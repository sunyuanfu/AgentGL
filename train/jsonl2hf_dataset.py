import argparse
from datasets import load_dataset

def convert_jsonl_to_dataset(input_file: str, output_folder: str):
    """Convert a JSONL file into a HuggingFace dataset."""
    try:
        print(f"正在加载JSONL文件: {input_file}")
        
        # Load the JSONL file as a dataset.
        dataset = load_dataset("json", data_files=input_file)
        print(f"数据集信息: {dataset}")
        
        # Print a small preview of the dataset.
        print("\n数据集预览:")
        print(dataset["train"][:3])  # Show the first 3 samples.
        
        print(f"\n正在保存数据集到: {output_folder}")
        
        # Save the dataset to the target directory.
        dataset["train"].save_to_disk(output_folder)
        
        print(f"转换完成！数据集已保存到: {output_folder}")
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
    except Exception as e:
        print(f"转换过程中出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='将JSONL文件转换为HuggingFace数据集格式')
    parser.add_argument('--input', required=True, help='输入的JSONL文件路径')
    parser.add_argument('--output', required=True, help='输出文件夹路径')
    
    args = parser.parse_args()
    
    convert_jsonl_to_dataset(args.input, args.output)

if __name__ == "__main__":
    main()
