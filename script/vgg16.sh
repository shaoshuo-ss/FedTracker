python main.py \
    --epochs 300 \
    --num_clients 50 \
    --clients_percent 0.4 \
    --pre_train 'False' \
    --model 'VGG16' \
    --dataset "cifar10" \
    --num_classes 10 \
    --image_size 32 \
    --gpu 7 \
    --seed 1 \
    --save_dir "./result/VGG16/" \
    --embed_layer_names "model.bn8;model.bn9;model.bn10" \
    --lambda1 0.0005 \
    --lambda2 0.005 \
    --test_interval 10
