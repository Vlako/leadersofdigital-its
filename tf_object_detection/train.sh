source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/model_main.py --pipeline_config_path=mask_rcnn_inception_resnet_v2_atrous_coco.config --model_dir=mask_rcnn_inception_resnet_v2/  --sample_1_of_n_eval_examples=1 --alsologtostderr 
