from transformers import LlamaTokenizer

text = '''
    白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    好好好，这般的好；哈哈哈……
    黑龙江第十一辆比亚迪卖出——
    ```鮪鯀鯪鯽：~！￥（）【】、“”？—《》０１２３４５６７８９『』
    ### 上海市
    <table><tbody><tr><td></td></tr></tbody></table>
    The primary use of LLaMA is research on large language models...
    public static void main(String[] args) {
    \x80\u200e
    '''

if __name__ == '__main__':
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained('tokenizer')
    print(chinese_llama_tokenizer.tokenize(text))
