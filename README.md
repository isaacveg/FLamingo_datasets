# FLamingo_datasets

A collection of datasets for FLamingo framework.

## Distributions
- IID
- Class distribution skew, where clients have different classes of data
- label distribution skew, where clients have same classes but different label distribution
- Predefined partition matrix, shape: (num_classes, num_clients), sum of each line is 1

## Datasets
The usage of each dataset is similar.  
Examples belew are: --nc 30 clients, --dist distribution is iid, --blc balance clients' dataset sizes, --seed 2048, --cc each client has 5 classes, --alpha 0.1 for dirichlet distribution, --least_samples 10 minimum samples for generation, --indir ../datasets/ where original file locates, --outdir ../datasets/ where processed dataset stored.

- [x] CIFAR-10    
```bash
python cifar10.py --nc 30 --dist iid --blc 1 --seed 2048 --cc 5 --alpha 0.1 --least_samples 10 --indir ../datasets/ --outdir ../datasets/
```
- [ ] CIFAR-100
- [ ] MNIST
- [ ] Fashion-MNIST
- [ ] EMNIST
- [ ] Tiny-ImageNet
- And more...


## Contributing
Feel free to contribute to this repository by opening a pull request.    
Or requesting a new dataset by opening an issue.
