import ray
from vllm import LLM, SamplingParams


# Define a Ray remote function to run vLLM inference.
@ray.remote(num_gpus=1)
def generate_text(prompt):
    # Initialize a vLLM LLM object (swap in any model you want).
    llm = LLM(model="/PATH/TO/HOME/user/Qwen2-1.5B-Instruct")
    # Configure sampling parameters.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Generate text.
    result = llm.generate(prompt, sampling_params)
    return result[0].outputs[0].text


if __name__ == "__main__":
    try:
        # Connect to an existing Ray cluster.
        ray.init(address='auto')

        # Define multiple prompts to generate.
        prompts = [
            "介绍一下北京故宫",
            "简述太阳系的结构",
            "讲讲唐朝的文化"
        ]

        # Submit tasks to Ray in parallel.
        result_ids = [generate_text.remote(prompt) for prompt in prompts]

        # Fetch all task results.
        results = ray.get(result_ids)

        # Print the result of each task.
        for i, result in enumerate(results):
            print(f"提示 {i + 1} 生成的文本：")
            print(result)
            print("-" * 50)

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
    finally:
        # Shut down Ray.
        ray.shutdown()
