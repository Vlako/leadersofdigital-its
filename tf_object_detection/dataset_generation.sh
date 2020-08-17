source venv/bin/activate
rm data/train-*
python dataset_generation.py --subset='train'
rm data/val-*
python dataset_generation.py --subset='val'
rm data/full_meta-*
python dataset_generation.py --subset='full_meta'
