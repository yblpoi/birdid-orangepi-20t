"""Compare BirdTrace predictions: NPU (.om via ACL) vs ONNX Runtime CPU."""
import argparse, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from birdid.config import load_config
from birdid.runtime import build_runtime
from birdid.classifier import SpeciesRegistry
from birdid.detector import decode_predictions, pick_targets
from birdid.preprocess import letterbox, crop_with_margin, resize_for_classifier

def logits_of(rt, img):
    return np.asarray(rt.infer(img)[0], dtype=np.float32).reshape(-1)

def topk(logits, k=5):
    idx = np.argsort(-logits, kind='stable')[:k]
    return [(int(i), float(logits[i])) for i in idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--image', required=True)
    ap.add_argument('--rounds', type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_dir = cfg.resolve(cfg.npu.model_dir)
    reg = SpeciesRegistry(cfg.resolve(cfg.npu.classes_file))
    seg = build_runtime(model_dir, cfg.npu.seg_model, cfg.npu.seg_onnx, cfg.npu.device_id)
    img = Image.open(args.image).convert('RGB')
    lb = letterbox(img, cfg.npu.seg_input_size)
    raw = seg.infer(lb.image)[0]
    dets = decode_predictions(np.asarray(raw), lb, img.size, cfg.npu.seg_conf_threshold,
                              cfg.npu.seg_iou_threshold, cfg.npu.max_detections)
    birds, others = pick_targets(dets, cfg.npu.max_detections)
    print('dets=%d birds=%d others=%d' % (len(dets), len(birds), len(others)))
    if not birds:
        print('NO_BIRD')
        return 1
    box = list(birds[0].box)
    crop = crop_with_margin(img, box, cfg.npu.crop_margin_ratio)
    cls_in = resize_for_classifier(crop, cfg.npu.cls_input_size)
    npu = build_runtime(model_dir, cfg.npu.cls_model, cfg.npu.cls_onnx, cfg.npu.device_id)
    cpu = build_runtime(model_dir, None, cfg.npu.cls_onnx, 0)
    print('npu_backend=%s cpu_backend=%s' % (npu.name, cpu.name))
    ln = logits_of(npu, cls_in)
    lc = logits_of(cpu, cls_in)
    tn, tc = topk(ln), topk(lc)
    same = all(a[0] == b[0] for a, b in zip(tn, tc))
    print('npu_top5=%s' % [(reg.lookup(i)[1], round(v, 3)) for i, v in tn])
    print('cpu_top5=%s' % [(reg.lookup(i)[1], round(v, 3)) for i, v in tc])
    print('top1_match=%s all5_match=%s' % (tn[0][0] == tc[0][0], same))
    print('max_abs_diff=%.4f' % float(np.max(np.abs(ln - lc))))
    for name, rt in (('npu', npu), ('cpu', cpu)):
        logits_of(rt, cls_in)
        t0 = time.time()
        for _ in range(args.rounds):
            logits_of(rt, cls_in)
        print('%s_avg_ms=%.1f' % (name, (time.time() - t0) / args.rounds * 1000))
    npu.close()
    cpu.close()
    return 0

if __name__ == '__main__':
    sys.exit(main())