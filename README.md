# SketchCLIP
Do Generalised Classifiers really work on Human Drawn Sketches ? ECCV-2024

### Inference :
Create environment
```
conda env create -f environment.yml
```
Pull weights from huggingface: https://huggingface.co/Hmrishav/SketchCLIP/

Run interface with gradio: 
```
python app.py
```

### BibTeX: 
```
@inproceedings{bandyopadhyay2025generalised,
  title={Do Generalised Classifiers Really Work on Human Drawn Sketches?},
  author={Bandyopadhyay, Hmrishav and Chowdhury, Pinaki Nath and Sain, Aneeshan and Koley, Subhadeep and Xiang, Tao and Bhunia, Ayan Kumar and Song, Yi-Zhe},
  booktitle={ECCV},
  year={2025},
}
```

### Code Acknowledgement:
- [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)
- [CoOp + CoCoOp](https://github.com/KaiyangZhou/CoOp)
