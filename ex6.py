import pandas as pd
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric


dataset = StandardDataset(
    df=pd.read_csv('adult.csv'),
    label_name='income',
    favorable_classes=['>50K', '>50k'],
    protected_attribute_names=['sex'],
    privileged_classes=[['Male']],
    metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
              'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]}
)


privileged_groups = [{'sex': 1}]   
unprivileged_groups = [{'sex': 0}] 

metric_dataset = BinaryLabelDatasetMetric(
    dataset, 
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("Statistical Parity Difference:", metric_dataset.statistical_parity_difference())