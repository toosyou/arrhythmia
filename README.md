# Arrhythmia heartbeat 
This is a project that can classify the heartbeat into N,S,V,F,Q class from a EKG signal.

It base on the AAMI heartbeat class.


![](https://i.imgur.com/4RB5YUB.png)


## Data preprocessing

### Data seperation
This project is using MIT-BIH database, which have 44 records from 43 subjects. (Records 201 and 202 came from the same male subject). Each has one signal, and the sampling rate is 360Hz.

First, take off Records 102, 104, 107 and 217, which almost only have Paced beats. Also, remove the Records 231 which doesn't have Normal beats.

After taken off the unusable signal. Split the 39 records into training set, validation set and testing set, each has 29 records, 6 records and 8 records  respectively.

The spliting way make sure that each class will have the close ratio in each set.


### Signal preprocessing process:
1. do the baseline wander removal, and use wavelet threshold to denoise signals.
2. cut each signal into the certain size.
3. downsampling to 250Hz.

### Evaluate

Use the Method applied from the  document `IEC 60601-2-47` , performing the beat-by-beat comparison.



# Usage
## Dependencies
1. Use `pipenv` to install dependencies and activate virtualenv:
    ```
    pipenv sync
    ```
2. `wandb` setup:
    ```
    wandb login
    ```
## Data Preprocessing

Setup `config.ini`

## Training
```
pipenv run python3 train.py
```
## Prediction
See [`tutorial.ipynb`](./tutorial.ipynb) for more details about prediction.





