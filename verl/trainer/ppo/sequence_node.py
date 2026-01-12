import torch
import numpy as np
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.utils.torch_functional import pad_sequence_to_length

def check_step_token(response_ids, bos_step_token_id, eos_step_token_id):
    if response_ids.count(bos_step_token_id) == 1 and response_ids.count(eos_step_token_id) == 1:
        return True
    return False


def check_step_format(response_ids, bos_step_token_id, eos_step_token_id, max_step_summary_length):
    step_token_format = check_step_token(response_ids, bos_step_token_id, eos_step_token_id)
    if not step_token_format:
        return False

    # check summary content length
    bos_pos = response_ids.index(bos_step_token_id)
    eos_pos = response_ids.index(eos_step_token_id)
    summary_length = eos_pos - bos_pos - 1
    if summary_length > max_step_summary_length:
        return False

    return True


def think_end_in_res(response_str):
    if "</think>" in response_str:
        return True
    return False


def extract_input_ids(input_ids, attention_mask):
    """
    attn: [0,0,0,1,1,1,1,0,0,0]
    input_ids: [10,11,12,13,14,15,16,17,18,19]
    -> [13,14,15,16]
    """
    assert len(input_ids) == len(attention_mask)
    result = [input_ids[i] for i in range(len(attention_mask)) if attention_mask[i] == 1]
    return result


def compute_leaf_adv(leaf_nodes):
    # compute mean reward per uid
    id2mean = {}
    for node in leaf_nodes:
        node_id = node.data_dict["uid"]
        if node_id not in id2mean:
            id2mean[node_id] = []
        id2mean[node_id].append(node.reward)
    for uid in id2mean:
        id2mean[uid] = sum(id2mean[uid]) / len(id2mean[uid])
    
    # assign advantage
    for node in leaf_nodes:
        node_id = node.data_dict["uid"]
        mean_reward = id2mean[node_id]
        node.advantage = node.reward - mean_reward

def convert_batch_to_lst(batch: DataProto):
    data_dict_lst = []

    # batch tensor
    for i in range(batch.batch.batch_size[0]):
        single_data_dict = {key: batch.batch[key][i : i + 1] for key in batch.batch.keys()}
        data_dict_lst.append(single_data_dict)

    # nonbatch tensor
    for i, data_dict in enumerate(data_dict_lst):
        for key in batch.non_tensor_batch.keys():
            data_dict[key] = batch.non_tensor_batch[key][i]

    return data_dict_lst
    
def convert_lst_to_batch(data_dict_lst: list[dict]):
    batch_dict = {}
    non_tensor_batch_dict = {}

    # batch tensor
    for key in data_dict_lst[0].keys():
        if isinstance(data_dict_lst[0][key], torch.Tensor):
            batch_dict[key] = torch.cat([data_dict[key] for data_dict in data_dict_lst], dim=0)

    # non-batch tensor
    for key in data_dict_lst[0].keys():
        if key == "response_str":
            continue
        if not isinstance(data_dict_lst[0][key], torch.Tensor):
            non_tensor_batch_dict[key] = [data_dict[key] for data_dict in data_dict_lst]

    # combine
    combine_batch_dict = {}
    combine_batch_dict.update(batch_dict)
    combine_batch_dict.update(non_tensor_batch_dict)

    return DataProto.from_dict(combine_batch_dict)


def collect_train_batch(root_node_lst, chunk_size):
    node_batch_lst = []
    for root in root_node_lst:
        current_node = root
        while len(current_node.children) != 0:
            current_node = current_node.children[0]
            node_batch_lst.append(current_node.format_batch())
    
    pad_size = chunk_size - len(node_batch_lst) % chunk_size
    for _ in range(pad_size):
        pad_node = root_node_lst[0].children[0]
        node_batch_lst.append(current_node.format_pad_batch())

    train_batch = DataProto.concat(node_batch_lst)
    return train_batch


def recurse_sequence_node_adv(node):
    if len(node.children) == 0:
        # leaf node
        return node.advantage, node.reward

    assert len(node.children) == 1
    child_adv, child_reward = recurse_sequence_node_adv(node.children[0])
    node.advantage = child_adv
    node.reward = child_reward
    return node.advantage, node.reward


def sequence_broadcast_adv(nodes_list, do_print=False):
    for node in nodes_list[0]:
        recurse_sequence_node_adv(node)

    if do_print:
        for node in nodes_list[0]:
            current_node = node
            while len(current_node.children) != 0:
                current_node = current_node.children[0]
                print(current_node.advantage, end=" ")
            print("\n--------------")


class SequenceNode:
    def __init__(self, data_dict, bos_step_token_id, eos_step_token_id, pad_token_id, replace_ids, depth=0, prompt_length=10240, max_response_length=4096, max_step_summary_length=1536):
        self.depth = depth
        self.prompt_length = prompt_length
        self.max_response_length = max_response_length
        self.max_step_summary_length = max_step_summary_length

        self.reward = None
        self.advantage = None
        self.children = []

        self.data_dict = data_dict
        self.bos_step_token_id = bos_step_token_id
        self.eos_step_token_id = eos_step_token_id
        self.pad_token_id = pad_token_id
        self.replace_ids = replace_ids
        
        # if self.depth == 0:
        #     self.is_end = False # 记录是否为叶子节点，只有当输出包含 '</think>' 或者 不符合step format时, 才为True
        #     self.think_end_in_res = False # 记录response中是否包含'</think>'
        #     self.step_format = True # 记录response 是否符合 step format: 包含俩step tokens，且summary content长度不超过max_step_summary_length
        # else:
        #     self.step_format = check_step_format(self.data_dict["responses"][0].tolist(), self.bos_step_token_id, self.eos_step_token_id, self.max_step_summary_length)
        #     self.think_end_in_res = think_end_in_res(self.data_dict["response_str"][0])

        #     if self.think_end_in_res:
        #         self.is_end = True
        #     elif not self.step_format:
        #         self.is_end = True
        #         self.reward = 0
        #     else:
        #         self.is_end = False

        if self.depth == 0:
            self.is_end = False # 记录是否为叶子节点，只有当输出包含 '</think>' 或者 不符合step format时, 才为True
            self.step_format = True # 记录response 是否符合 step format: 包含俩step tokens，且summary content长度不超过max_step_summary_length
        else:
            self.step_format = check_step_format(self.data_dict["responses"][0].tolist(), self.bos_step_token_id, self.eos_step_token_id, self.max_step_summary_length)

            if not self.step_format:
                self.is_end = True
            else:
                self.is_end = False
    

    def format_reward_tensor(self):
        assert self.reward is not None
        reward_tensor = torch.zeros_like(self.data_dict["responses"], dtype=torch.float32)
        valid_response_length = self.data_dict["attention_mask"][:, self.prompt_length:].sum(dim=-1)
        reward_tensor[0, valid_response_length[0].item() - 1] = self.reward
        self.reward_tensor = reward_tensor


    def format_attn_pos(self, input_ids):
        # attention_mask: [1,......]
        # position_ids: [0, 1, 2,......]
        attention_mask = np.array([1] * len(input_ids), dtype=np.int64)
        position_ids = np.array(list(range(len(input_ids))), dtype=np.int64)
        # convert to list
        attention_mask = attention_mask.tolist()
        position_ids = position_ids.tolist()
        return attention_mask, position_ids


    def format_fold_input_ids(self):
        input_part_ids = self.data_dict["input_ids"][0, :self.prompt_length].tolist()
        response_part_ids = self.data_dict["input_ids"][0, self.prompt_length:].tolist()
        input_part_attnention_mask = self.data_dict["attention_mask"][0, :self.prompt_length].tolist()
        response_part_attention_mask = self.data_dict["attention_mask"][0, self.prompt_length:].tolist()

        # extract ids
        input_part_ids_lst = extract_input_ids(input_part_ids, input_part_attnention_mask)
        response_part_ids_lst = extract_input_ids(response_part_ids, response_part_attention_mask)

        # split from self.bos_step_token_id pos in response_part_ids_lst
        split_pos = response_part_ids_lst.index(self.bos_step_token_id)
        response_part_ids_lst = response_part_ids_lst[split_pos:]
        response_part_ids_lst = self.replace_ids + response_part_ids_lst

        # concate to new raw_prompt_ids
        raw_prompt_ids = input_part_ids_lst + response_part_ids_lst

        return raw_prompt_ids


    def left_pad_and_convert_to_tensor(self, input_list, target_length, pad_id):
        # 计算需要填充的长度
        pad_length = target_length - len(input_list)
        
        # 左填充：在列表前面添加pad_length个pad_id
        if pad_length > 0:
            padded_list = [pad_id] * pad_length + input_list
        else:
            # 如果列表已经比目标长度长，可以截断（根据需求调整）
            padded_list = input_list[-target_length:]  # 保留最后target_length个元素

        # 转换为张量并调整形状
        tensor = torch.tensor(padded_list, dtype=torch.int64)
        tensor = tensor.unsqueeze(0)  # 添加批次维度，形状变为 [1, target_length]
        
        return tensor

    def format_pad_batch(self):
        assert self.advantage is not None
        assert self.depth != 0

        node_batch: DataProto = DataProto.from_single_dict({
            'input_ids': self.data_dict["input_ids"],
            'attention_mask': self.data_dict["attention_mask"],
            'response_mask': self.data_dict["attention_mask"][:, -self.max_response_length:],
            'position_ids': self.data_dict["position_ids"],
            'prompts': self.data_dict["prompts"],
            'responses': self.data_dict["responses"],
            'data_source': np.array([self.data_dict['data_source']], dtype=object),
            'reward_key': np.array([self.data_dict['reward_key']], dtype=object),
            'ability': np.array([self.data_dict['ability']], dtype=object),
            'extra_info': np.array([self.data_dict['extra_info']], dtype=object),
            'index': np.array([self.data_dict['index']], dtype=object),
            'reward_model': np.array([self.data_dict["reward_model"]], dtype=object),
            'tools_kwargs': np.array([self.data_dict['tools_kwargs']], dtype=object),
            'interaction_kwargs': np.array([self.data_dict['interaction_kwargs']], dtype=object),
            'uid': np.array(["pad"], dtype=object),
            'sid': np.array(["pad"], dtype=object),
        })

        acc = torch.tensor([0], dtype=torch.float32, device='cpu')
        token_level_rewards = torch.zeros_like(self.data_dict["responses"], dtype=torch.float32)
        advantage_tensor = torch.zeros_like(self.data_dict["responses"], dtype=torch.float32)
        
        node_batch.batch["acc"] = acc
        node_batch.batch["token_level_rewards"] = token_level_rewards
        node_batch.batch["token_level_scores"] = token_level_rewards
        node_batch.batch["advantages"] = advantage_tensor
        node_batch.batch["returns"] = advantage_tensor

        node_batch.batch["response_mask"] = 0 * node_batch.batch["response_mask"]
        node_batch.batch["attention_mask"][:, -self.max_response_length:] = 0
        return node_batch



    def format_batch(self):
        assert self.depth != 0

        node_batch: DataProto = DataProto.from_single_dict({
            'input_ids': self.data_dict["input_ids"],
            'attention_mask': self.data_dict["attention_mask"],
            'response_mask': self.data_dict["attention_mask"][:, -self.max_response_length:],
            'position_ids': self.data_dict["position_ids"],
            'prompts': self.data_dict["prompts"],
            'responses': self.data_dict["responses"],
            'data_source': np.array([self.data_dict['data_source']], dtype=object),
            'reward_key': np.array([self.data_dict['reward_key']], dtype=object),
            'ability': np.array([self.data_dict['ability']], dtype=object),
            'extra_info': np.array([self.data_dict['extra_info']], dtype=object),
            'index': np.array([self.data_dict['index']], dtype=object),
            'reward_model': np.array([self.data_dict["reward_model"]], dtype=object),
            'tools_kwargs': np.array([self.data_dict['tools_kwargs']], dtype=object),
            'interaction_kwargs': np.array([self.data_dict['interaction_kwargs']], dtype=object),
            'uid': np.array([self.data_dict['uid']], dtype=object),
            'sid': np.array([self.data_dict['sid']], dtype=object),
        })
    
        if self.advantage is not None:
            acc = torch.tensor([self.reward], dtype=torch.float32, device='cpu')
            token_level_rewards = torch.zeros_like(self.data_dict["responses"], dtype=torch.float32)
            valid_response_length = self.data_dict["attention_mask"][:, self.prompt_length:].sum(dim=-1)
            token_level_rewards[0, valid_response_length[0].item() - 1] = self.reward

            advantage_tensor = torch.zeros_like(self.data_dict["responses"], dtype=torch.float32)
            advantage_tensor = advantage_tensor.fill_(self.advantage)
            advantage_tensor = advantage_tensor * node_batch.batch["response_mask"]
            
            node_batch.batch["acc"] = acc
            node_batch.batch["token_level_rewards"] = token_level_rewards
            node_batch.batch["token_level_scores"] = token_level_rewards
            node_batch.batch["advantages"] = advantage_tensor
            node_batch.batch["returns"] = advantage_tensor

        return node_batch


    def format_gen_batch_sample(self):
        if self.depth == 0:
            node_input_batch: DataProto = DataProto.from_single_dict({
                'input_ids': self.data_dict["input_ids"],
                'attention_mask': self.data_dict["attention_mask"],
                'position_ids': self.data_dict["position_ids"],
                'data_source': np.array([self.data_dict['data_source']], dtype=object),
                'reward_key': np.array([self.data_dict['reward_key']], dtype=object),
                'ability': np.array([self.data_dict['ability']], dtype=object),
                'extra_info': np.array([self.data_dict['extra_info']], dtype=object),
                'index': np.array([self.data_dict['index']], dtype=object),
                'reward_model': np.array([self.data_dict["reward_model"]], dtype=object),
                'raw_prompt_ids': np.array([self.data_dict['raw_prompt_ids']], dtype=object),
                'tools_kwargs': np.array([self.data_dict['tools_kwargs']], dtype=object),
                'interaction_kwargs': np.array([self.data_dict['interaction_kwargs']], dtype=object),
                'uid': np.array([self.data_dict['uid']], dtype=object),
                'sid': np.array([self.data_dict['sid']], dtype=object),
            })

        else:
            fold_input_ids = self.format_fold_input_ids()
            attention_mask, position_ids = self.format_attn_pos(fold_input_ids)
            # format to Tensor(shape=torch.Size([1, prompt_length]), device=cpu, dtype=torch.int64, is_shared=True)
            fold_input_ids_tensor = self.left_pad_and_convert_to_tensor(fold_input_ids, self.prompt_length, self.pad_token_id)
            attention_mask_tensor = self.left_pad_and_convert_to_tensor(attention_mask, self.prompt_length, 0)
            position_ids_tensor = self.left_pad_and_convert_to_tensor(position_ids, self.prompt_length, 0)
            node_input_batch: DataProto = DataProto.from_single_dict({
                'input_ids': fold_input_ids_tensor,
                'attention_mask': attention_mask_tensor,
                'response_mask': attention_mask_tensor[:, -self.max_response_length:],
                'position_ids': position_ids_tensor,
                'data_source': np.array([self.data_dict['data_source']], dtype=object),
                'reward_key': np.array([self.data_dict['reward_key']], dtype=object),
                'ability': np.array([self.data_dict['ability']], dtype=object),
                'extra_info': np.array([self.data_dict['extra_info']], dtype=object),
                'index': np.array([self.data_dict['index']], dtype=object),
                'reward_model': np.array([self.data_dict["reward_model"]], dtype=object),
                'raw_prompt_ids': np.array([fold_input_ids], dtype=object),
                'tools_kwargs': np.array([self.data_dict['tools_kwargs']], dtype=object),
                'interaction_kwargs': np.array([self.data_dict['interaction_kwargs']], dtype=object),
                'uid': np.array([self.data_dict['uid']], dtype=object),
                'sid': np.array([self.data_dict['sid']], dtype=object),
            })
        
        return node_input_batch