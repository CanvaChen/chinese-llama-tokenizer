from transformers import LlamaTokenizer

text = '''
<s>    test    </s>
### 登鹳雀楼
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
白日は山に沿って尽き、黄河は海に流れ込む。千里の眺めを求めるなら、もう一層の楼に上がれ。
The sun beyond the mountain glows. The Yellow River seaward flows. You can enjoy a grander sight. By climbing to a greater height.
if __name__ == '__main__':
public static void main(String[] args) {
<body><h1>标题一</h1><h2>标题二</h2><p>段落</p></body>
cos2α=2cos²α-1
f(b)-f(a)=f′(ξ)(b-a),ξ∈(a,b)
Ⅳ. 2³ ∛8
...
'''

if __name__ == '__main__':
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained('tokenizer-human')
    print(chinese_llama_tokenizer.tokenize(text))
