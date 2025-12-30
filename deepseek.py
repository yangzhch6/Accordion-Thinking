import openai
import re
from copy import deepcopy

# 配置 DeepSeek API
client = openai.OpenAI(
    api_key="sk-5dfc94b018f145579b88834a8c87237c", 
    base_url="https://api.deepseek.com/beta"
)

SYSTEM_PROMPT_old = """You are an advanced reasoning engine capable of solving complex problems through recursive thinking.

GLOBAL RULES:
1. You must think step-by-step to solve the user's problem.
2. When you reach a solution, start the final answer with "## Final Solution".
3. If the problem is not yet fully solved but you have completed a reasoning step, start your output with "## Current Progress".

FOLDING MECHANISM:
- You are working under a strict token budget per step.
- If you realize you are running out of space or receive a truncation warning, you MUST summarize your current logic state.
- The summary should include:
  - Key premises established.
  - Intermediate conclusions derived.
  - The solving plan you current have.

Format your response exactly as:
<think>
[Your detailed reasoning here...]
</think>
## Current Progress
[Summary if interrupted]

[Output Content starting with ## Final Solution or ## Current Progress]"""

SYSTEM_PROMPT = """You are an iterative reasoning engine designed to solve complex math and logic problems over multiple steps. For a hard problem, do not rush. Solve one logical part per turn.

### OPERATIONAL RULES:
1. **Iterative Solving**: You will receive a problem or a "Progress Summary" from a previous step. Your task is to advance the reasoning from that state.
2. **State Management**: 
    - You act as a stateless worker. You must rely entirely on the provided summary to know values, defined variables, and proven lemmas.
    - Do not re-derive what has already been established in the summary.
    - Summary should be detailed. It should includ:
        * Any **key variables, quantities, or concepts** introduced or defined in this step, along with their meaning.
        * Any **assumptions or conditions** applied or established in this step.
        * The core **logical derivation or calculation** performed in this step.
        * The **specific conclusion, result, or output** of this step (e.g., a derived formula, an intermediate value, a decision point).
3. **Reasoning (Internal)**: Use your internal chain-of-thought to perform rigorous calculations and logical deductions.
4. **Budget**: For each step, the maximum reasoning token is limited to 4000.
5. **Output (External)**: Your final response content must be strictly formatted based on your status.

### OUTPUT FORMAT:
**Case A: Problem Solved**
If you have reached the final mathematical result:
## Final Solution
[Provide the direct answer and a very brief justification]

**Case B: Intermediate Step Completed**
If you have made progress (e.g., found a side length, established a geometric relationship) but the final answer is not reached yet:
## Current Progress
[List new values, relationships, and conclusions found in this step, this part should be **detailed**]"""

USER_QUESTION = "In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$."

user_link_prompt = "Review the 'Current Progress' above. Based on those established facts, execute the next logical step of the reasoning."

def run_cot_folding_agent(question):
    # 初始化多轮对话历史
    # 注意：为了折叠效果，我们实际上会动态维护这个列表
    # 这里的 history 只保留 [System, User(原始问题), Assistant(之前的Summary), User(Continue)...]
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    conversation_history_think = deepcopy(conversation_history)
    
    step_count = 0
    max_step_tokens = 4000
    
    while True:
        step_count += 1
        print(f"\n=== Step {step_count} Start ===")
        
        # 1. 调用 API
        current_response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=conversation_history,
            max_tokens=max_step_tokens,
            temperature=0.7,
            stop=None 
        )
        
        current_content = current_response.choices[0].message.content
        current_reasoning_content = current_response.choices[0].message.reasoning_content
        finish_reason = current_response.choices[0].finish_reason

        # Logic 2: 处理截断 (Token Limit Reached)
        if finish_reason == 'length':
            print(">>> Hit 4000 token limit. Injecting Summary Trigger...")
            
            # 你的策略：在截断处 append 文本，让模型补全 Summary
            append_text = "\n\n[SYSTEM: Length limit reached. Summarize current derived conclusions.]"
            step_reasoning_content = current_reasoning_content + append_text

            # 这里的技巧是：因为是截断，我们需要再调一次 API 来获得 Summary
            temp_history = deepcopy(conversation_history)
            temp_history.append({
                "role": "assistant", 
                "reasoning_content": step_reasoning_content,
                "content": "## Current Progress\n",
                "prefix": True
            })
            
            # 让模型生成 Summary 及其后的 Output
            summary_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=temp_history,
                max_tokens=1000, # 给足够空间写总结
                temperature=0.6
            )
            
            summary_content = "## Current Progress\n" + summary_response.choices[0].message.content
            print(f"--- Generated Summary ---\n{summary_content}...\n-------------------------")
            
            # Logic 3: 折叠过程 (Folding)
            # 构建折叠后的 Assistant 消息            
            conversation_history.append({"role": "assistant", "content": summary_content})
            conversation_history.append({"role": "user", "content": user_link_prompt})

            conversation_history_think.append({"role": "assistant", "content": summary_content, "reasoning_content": step_reasoning_content})
            conversation_history_think.append({"role": "user", "content": user_link_prompt})
            
        # Logic 2b: 模型自己停了 (Finish normally)
        else:
            print(">>> Step finished naturally.")
            
            # 检查是否包含最终解决方案
            if "## Final Solution" in current_content:
                print("\n🎉 PROBLEM SOLVED!")
                # 提取最终答案
                final_ans = current_content.split("## Final Solution")[1]
                print(final_ans.strip())
                break
            
            elif "## Current Progress" in current_content:
                print(">>> Intermediate step done. Folding context...")        
                conversation_history.append({"role": "assistant", "content": current_content})
                conversation_history.append({"role": "user", "content": user_link_prompt})

                conversation_history_think.append({"role": "assistant", "content": current_content, "reasoning_content": current_reasoning_content})
                conversation_history_think.append({"role": "user", "content": user_link_prompt})
            
            else:
                # 异常情况处理：模型既没说结束也没说进行中，强制停止
                break

# 运行
run_cot_folding_agent(USER_QUESTION)