

# train the entire model.
export DATA_ROOT=${GEMINI_DATA_IN1}/COCO2014 
export DATA_ROOT2=${GEMINI_DATA_IN2}            
export OUTPUT_ROOT=${GEMINI_DATA_OUT}
export MODEL_ROOT1=${GEMINI_PRETRAIN}
export MODEL_ROOT2=${GEMINI_PRETRAIN2}
python train_caption.py exp.name=caption_finetune_region \
    model.detector.checkpoint=${MODEL_ROOT1}/detector_checkpoint.pth \
    optimizer.finetune_xe_epochs=10 \
    optimizer.finetune_sc_epochs=10 \
    optimizer.batch_size=32 \
    optimizer.num_workers=4 \
    exp.ngpus_per_node=8 \
    exp.world_size=8 \
    model.cap_generator.decoder_name=Parallel \
    dataset.overfit=False \
    exp.start_epoch=10

