# Fine-Tuning

## File Structure

    .
    ├── BERT.py                         # BERT class
    ├── ClassificationDataset.py        # dataset class
    ├── README.md                       # README file
    ├── app.py                          # RESTful API server
    ├── data.py                         # fetch and preprocess datasets
    ├── data_visualization.py           # visualize model ouputs, plot loss curves
    ├── finetune.py                     # finetune model
    ├── index.html                      # page for API
    ├── pretrain.py                     # further pretrain model
    ├── requirements.txt                # specify what python packages are required to run the project 
    ├── test.py                         # test model
    └── utilities.py                    # utility functions

## How To Run

## Performance Evaluation

|  | YelpReviewPolarity | AmazonReviewFull | IMDB |
| --- | --- | --- | --- |
| None | 0.906289474 | 0.507292308 | 0.8578 |
| YelpReviewPolarity | 0.973921053 | 0.583553846 | 0.92456 |
| AmazonReviewFull | 0.957789474 | **0.637030769** | 0.9404 |
| IMDB | 0.930631579 | 0.541938462 | 0.93404 |
| All except Yelp | 0.950105263 | - | - |
| All except Amazon | - | 0.583692308 | - |
| All except IMDB | - | - | 0.937 |
| All| **0.975394737** | 0.625107692 | **0.94276** |

The table above shows the best test accuracy reached with different further pre-train datasets. The first row is the
test dataset, and the first columns is the pre-train datasets. We discovered that different datasets in the same task
help create a more generalized model that has decent performance on more than one dataset.
