#template vicuna, llama2, llama3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


#for i in {1..7}
for i in {7..7}
do
	llamafactory-cli train config_predict/config_predict_gsm8k.yaml
	echo "predict done"
	sleep 60s

done
