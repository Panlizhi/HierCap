
#  MSC0C0 Karpathy split evaluation
export DATA_ROOT=${GEMINI_DATA_IN1}/COCO2014 
export DATA_ROOT2=${GEMINI_DATA_IN2}            
export OUTPUT_ROOT=${GEMINI_DATA_OUT}
export MODEL_ROOT1=${GEMINI_PRETRAIN}
export MODEL_ROOT2=${GEMINI_PRETRAIN2}
python eval_caption.py +split='test' ++model.cap_generator.decoder_name=Concatenated_Parallel \
    exp.checkpoint=/gemini/pretrain2/checkpoint_best_valid.pth




