use ffnn::activations::*;
use ffnn::NetBuilder;

use ndarray::prelude::*;

use plotters::prelude::*;
use polars::prelude::*;

fn main() {
    let mut net = NetBuilder::new(1).layer(1, LINEAR).init();

    // Read dataset
    let df = CsvReader::from_path("examples/train.csv")
        .unwrap()
        .finish()
        .unwrap();

    let x = df
        .select(["x"])
        .unwrap()
        .to_ndarray::<Float32Type>()
        .unwrap();
    let y = df
        .select(["y"])
        .unwrap()
        .to_ndarray::<Float32Type>()
        .unwrap();

    println!("Initial error: {}", net.mse(&x, &y));
    println!("training . . . ");
    let costs = net.train(&x, &y, 100000, 2, 0.001);
    println!("Error after training: {}", net.mse(&x, &y));

    let m = net.layers[0].weights[[0, 0]];
    let b = net.layers[0].biases[0];

    println!("y = {}x + {}", m, b);

    // Graph costs
    let root = BitMapBackend::new("plotters-doc-data/0.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..1000f32, 0f32..500f32)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            costs.iter().enumerate().map(|(i, c)| (i as f32, *c)),
            &RED,
        ))
        .unwrap()
        .label("cost");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
}
