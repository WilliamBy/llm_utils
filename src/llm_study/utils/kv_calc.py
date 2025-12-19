import argparse
from transformers import AutoConfig

def calculate_kv_cache_size(model_id: str, precision_bytes: float, precision_name: str):
    """
    自动获取模型配置并计算每token的KV Cache大小。
    Args:
        model_id (str): Hugging Face Hub上的模型ID。
        precision_bytes (float): 每个参数占用的字节数 (支持小数, 如INT4为0.5)。
        precision_name (str): 精度的名称，用于打印 (e.g., "fp16")。
    """
    try:
        print(f"正在从 Hugging Face Hub 获取 '{model_id}' 的配置...")
        config = AutoConfig.from_pretrained(model_id)

        # --- 1. 提取所需参数 ---
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        num_q_heads = config.num_attention_heads
        
        # 鲁棒地获取K/V头的数量，以兼容标准MHA、GQA和MQA
        # 如果 num_key_value_heads 不存在，则说明是标准多头注意力(MHA)，K/V头数量等于Q头数量
        num_kv_heads = getattr(config, 'num_key_value_heads', num_q_heads)

        # --- 2. 计算派生参数 ---
        head_dim = hidden_size // num_q_heads

        # --- 3. 计算KV Cache大小 ---
        # 公式: (层数) * (每个token在单层产生的KV向量总大小)
        # 单层大小: (K/V头数量) * (每个头的维度) * 2 (K和V) * (每个数值的字节数)
        size_per_token_bytes = num_layers * num_kv_heads * head_dim * 2 * precision_bytes
        
        # 转换为KB
        size_per_token_kb = size_per_token_bytes / 1024

        # --- 4. 打印详细报告 ---
        print("\n" + "="*50)
        print(f"KV Cache 计算报告 for: {model_id}")
        print("="*50)
        print(f"数据精度 (Precision):       {precision_name.upper()} ({precision_bytes} bytes/value)")
        print("-" * 50)
        print(f"模型架构参数:")
        print(f"  - 模型层数 (L):             {num_layers}")
        print(f"  - 隐藏层维度 (Hidden Size):   {hidden_size}")
        print(f"  - Query 头数量 (N_q):       {num_q_heads}")
        print(f"  - Key/Value 头数量 (N_kv):  {num_kv_heads} ({'GQA/MQA' if num_kv_heads < num_q_heads else 'MHA'})")
        print(f"  - 每个头的维度 (d):         {head_dim}")
        print("-" * 50)
        print("计算过程:")
        print(f"  公式: L * N_kv * d * 2 * bytes_per_value")
        print(f"  代入: {num_layers} * {num_kv_heads} * {head_dim} * 2 * {precision_bytes} = {size_per_token_bytes:,.0f} 字节")
        print("-" * 50)
        print(f"结果: 每 Token 的 KV Cache 大小 = {size_per_token_kb:.2f} KB")
        print("="*50)
        
        # --- 5. 打印实际应用示例 ---
        print("\n示例: 不同序列长度下的总KV Cache显存占用:")
        for seq_len_k in [1, 8, 32, 128]:
            seq_len = seq_len_k * 1024
            total_mb = (seq_len * size_per_token_kb) / 1024
            total_gb = total_mb / 1024
            print(f"  - {seq_len_k: >3}K tokens ({seq_len:,} tokens): {total_mb: >8,.2f} MB ({total_gb:.2f} GB)")

    except Exception as e:
        print(f"\n[错误] 计算失败: {e}")
        print("请检查模型ID是否正确，以及你的网络连接。")

def main():
    parser = argparse.ArgumentParser(
        description="计算给定 Hugging Face 模型每 token 产生的 KV Cache 大小。",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息格式
    )
    
    parser.add_argument(
        "model_id",
        type=str,
        help="要查询的 Hugging Face Hub 模型ID。\n例如: 'meta-llama/Llama-3.1-8B-Instruct' 或 'gpt2'"
    )
    
    parser.add_argument(
        "-p", "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16", "int8", "int4"],
        help="用于存储KV Cache的数值精度。\n"
             "  - fp32: 4 字节\n"
             "  - fp16/bf16: 2 字节 (默认)\n"
             "  - int8: 1 字节\n"
             "  - int4: 0.5 字节"
    )

    args = parser.parse_args()
    
    # 将精度字符串映射到字节大小
    precision_map = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }
    
    bytes_per_value = precision_map[args.precision]
    
    calculate_kv_cache_size(args.model_id, bytes_per_value, args.precision)

if __name__ == "__main__":
    main()
