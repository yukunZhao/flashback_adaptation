export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for name in "config_train/config_train_MT.yaml"
do
	echo "Begin train $name"
	FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NOD=8 MASTER_ADDR=10.223.10.139 MASTER_PORT=29501 llamafactory-cli train $name
	echo "Train $name task done"
	sleep 200s
done

