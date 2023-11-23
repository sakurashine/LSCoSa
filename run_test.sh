array1=("1")  
array2=( "1_Dy4hot+1.6KL"  )
array3=( 0.4  )
length=${#array1[@]}
for ((i=0;i<$length;i++))
do
python linear.py   \
    -a resnet18 \
    --lr 2 \
    --batch_size 64 \
    --pretrained [your_checkpoint_path] \
    --epochs 200 \
    --folder "../dataset/IndianPines" \
    --dataset "IndianPines" \
    --patch_size 15 \
    --class_balancing  \
    --load_data 0.10  \
    --fine_tune no  \
    --desc ${array2[$i]} \
    --rho ${array3[$i]} \
    --norm 2
done
