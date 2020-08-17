source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

CHECKPOINT=$(grep -o -m 1 "model.ckpt-[0-9]*" mask_rcnn_inception_resnet_v2/checkpoint)

python object_detection/export_inference_graph.py --pipeline_config_path=mask_rcnn_inception_resnet_v2_atrous_coco.config --trained_checkpoint_prefix=mask_rcnn_inception_resnet_v2/$CHECKPOINT --output_directory=../mask_rcnn_inception_resnet_v2
 
