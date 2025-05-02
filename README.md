## Unitag

### Project Structure

```
├── graph_stage1/           # stage1_code
│   ├── config/
│   ├── models/   
├── graphvq /               # Stage2_code
│   ├── models/
│   ├── ds_stage2.py(under constructing)
│   ├── stage2_base.py(simple training procedure without any DDP/Deepspeed)
├── data/                   # data
├── evaluate/               # TBD
```


### Usage

For stage1:
deepspeed --num_gpus=2 new_stage1_training.py \
 --lm_model="/fs-computility/mabasic/shared/models/Qwen2.5-7B-Instruct" \
  --batch_size=1 \
  --num_epochs=1 \
  --learning_rate=2e-3 \
  --bf16 \
  --freeze_backbone \
  --deepspeed_config="zero2.json" \
  --max_seq_length=256  \
  --use_wandb \

For stage 2:\
python stage2_base.py --use_wandb --dataset cora --batch_size 16 --epochs 100 --lr 0.001 --train_llm --llm_epochs 30 --llm_lr 0.0001

For stage 2 with DDP:\
python -m torch.distributed.launch --nproc_per_node=4 stage2_ddp.py \
    --dataset cora \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.001 \
    --train_llm \
    --llm_epochs 30 \
    --llm_lr 0.0001 \
    --output_dir ./models/ddp_run \
    --use_wandb