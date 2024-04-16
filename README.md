# MMCode: Evaluating Multi-Modal Code Large Language Models with Visually Rich Programming Problems
<div style="text-align: center; font-size:14pt">
    <a href="#">Paper</a> | 
    <a href="https://huggingface.co/datasets/likaixin/MMCode">Huggingface Dataset</a>
</div>

## Dataset Description
MMCode is a multi-modal code generation dataset designed to evaluate the problem-solving skills of code language models in visually rich contexts. 
It contains 3,548 questions paired with 6,622 images, derived from real-world programming challenges across 10 code competition websites, with Python solutions and tests provided. 
The dataset emphasizes the extreme demand for reasoning abilities, the interwoven nature of textual and visual contents, and the occurrence of questions containing multiple images.

**For more detailed introduction of the data, please see the [🤗Huggingface Dataset Page](https://huggingface.co/datasets/likaixin/MMCode).**
# Usage
## Inference
Please configure API keys before running the models. They can be set in environment variables `OPENAI_API_KEY` and `GOOGLE_API_KEY`.

An example for GPT-4V generation:
```python
python generate.py \
    --model gpt4v \
    --problems_root <path_to_the_test_set> \
    --save_path "results/gpt4v-mmcode_test.jsonl"
```

## Evaluation
To evaluate the results generated by GPT-4V, run:
```python
python eval.py \
    --problems_root <path_to_the_test_set> \
    --generation_file "results/gpt4v-mmcode_test.jsonl"
```

# Citation
Please cite our work if you find it useful:
```plain
@misc{li2024mmcode,
      title={MMCode: Evaluating Multi-Modal Code Large Language Models with Visually Rich Programming Problems}, 
      author={Kaixin Li and Yuchen Tian and Qisheng Hu and Ziyang Luo and Jing Ma},
      year={2024},
      eprint={2404.09486},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```