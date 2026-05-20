mod knn;
mod random_forest;
mod svm;

pub use knn::{
    DistanceMetric, KNN, KnnConfig, accuracy, confusion_matrix_binary, sensitivity_from_cm,
    specificity_from_cm,
};
pub use random_forest::RandomForestWrapper;

#[allow(unused_imports)]
pub use svm::SVM;
