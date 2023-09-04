#!/usr/bin/env bash
# echo [STARTING] Timegan Sine Data Generation
# python main_timegan.py --timegan ori --batch_size 128 --data_name sine --output timegan/sine --seq_len 100 --module gru --iteration 10000 --metric_iteration 5 
# echo [FINISHED] Timegan Sine Data Generation

# echo [STARTING] TTGAN Sine Data Generation
# python main_timegan.py --timegan ttgan --batch_size 256 --data_name sine --output ttgan/sine --seq_len 100 --module gru --iteration 10000 --metric_iteration 5 
# echo [FINISHED] TTGAN Sine Data Generation

# echo [STARTING] TTGAN PSF Data Generation
# python main_timegan.py --timegan ttgan --batch_size 256 --data_name PSF --output ttgan/PSF/random --actuation rand_free_30 --seq_len 100 --module gru --iteration 10000 --metric_iteration 5 
# echo [FINISHED] TTGAN PSF Data Generation

echo [STARTING] TTGAN PSG Data Generation
python main_timegan.py --timegan ttgan --batch_size 256 --data_name PSG --output ttgan/PSG/random --actuation Random --seq_len 100 --module gru --iteration 10000 --metric_iteration 5 
echo [FINISHED] TTGAN PSG Data Generation

echo [STARTING] TTGAN PSC Data Generation
python main_timegan.py --timegan ttgan --batch_size 256 --data_name PSC --output ttgan/PSC --seq_len 100 --module gru --iteration 10000 --metric_iteration 5 
echo [FINISHED] TTGAN PSC Data Generation


