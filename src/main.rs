use std::{
    fs::File,
    str::FromStr,
    time::Instant,
    io::{self, BufRead, BufReader},
    path::Path,
};

use ndarray::{Axis, Array1, Array2, s};

fn main() {

    // read dataset
    let dataset: Array2<f64> = read_dataset("ex1data2.txt");
    let learning_rate:f64 = 0.01;
    let number_of_iterations:i32 = 500;

    let number_of_features = dataset.shape()[1] -1;
    let number_of_examples =  dataset.shape()[0];

    // create features matrix and fill it from dataset.
    let mut features: Array2<f64> = unsafe { Array2::uninitialized((number_of_examples, number_of_features))};
    features.assign(&dataset.slice(s![.., 0..-1]));
    let features_normalized = feature_normalization(features);

    // create targets matrix
    let mut targets: Array2<f64> = unsafe { Array2::uninitialized((number_of_examples, 1))};
    targets.slice_mut(s![.., 0]).assign(&dataset.slice(s![.., -1]));

    // add ones column to features for theta 0 param.
    let mut input_variables: Array2<f64> = Array2::ones((number_of_examples, number_of_features + 1));
    input_variables.slice_mut(s![.., 1..]).assign(&features_normalized);

    // learn theta parameters with gradient descent algorithm.
    let theta = gradient_descent(input_variables, targets, learning_rate, number_of_iterations);
    println!("{:?}", theta);
}

fn lines_from_file(filename: impl AsRef<Path>) -> io::Result<Vec<String>> {
    BufReader::new(File::open(filename)?).lines().collect()
}

fn read_dataset(filename: &str) ->Array2<f64> {
    let lines = lines_from_file(filename).expect("Could not load lines");

    let mut data = Vec::new();
    let mut number_of_features = 0;
    let mut number_of_examples = 0;
    for line in lines {
        let row: Vec<f64> = line.split(",").map(|s| f64::from_str(s).unwrap()).collect();
        number_of_features = row.len() - 1;
        data.extend_from_slice(&row);
        number_of_examples += 1;
    }

    let dataset: Array2<f64> = Array2::from_shape_vec((number_of_examples, number_of_features + 1), data).unwrap();
    dataset
}

fn gradient_descent(input_variables: Array2<f64>, targets: Array2<f64>, learning_rate: f64, number_of_iterations: i32) -> Array2<f64> {

    let input_variables_copy: Array2<f64> = input_variables.clone();

    let input_variables_transpose: Array2<f64> = input_variables_copy.reversed_axes();
    let mut predictions: Array2<f64>;
    let mut difference: Array2<f64>;
    let mut error: Array2<f64>;

    let number_of_examples: f64 = input_variables.shape()[0] as f64;
    let number_of_input_variables = input_variables.shape()[1];

    println!("number_of_examples: {}", number_of_examples);
    println!("number_of_input_variables: {}", number_of_input_variables);

    let learning_coefficient = learning_rate / number_of_examples;

    let mut theta: Array2<f64> = Array2::<f64>::zeros((number_of_input_variables, 1));

    let now = Instant::now();
    let mut i = 0;
    while i <number_of_iterations {
        predictions = input_variables.dot(&theta);
        difference = &predictions - &targets;
        error = input_variables_transpose.dot(&difference);
        theta = theta - error * learning_coefficient;
        i = i + 1;
    }

    println!("Training Time: {} ms", now.elapsed().as_millis());

    theta
}

fn feature_normalization(features: Array2<f64>) -> Array2<f64> {
    let mean: Array1<f64> = features.mean_axis(Axis(0)).unwrap();
    let std_dev: Array1<f64> = features.std_axis(Axis(0), 0.0);

    let features_normalized: Array2<f64> = (features - mean) / std_dev;
    features_normalized
}
