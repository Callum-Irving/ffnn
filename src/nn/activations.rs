use super::Float;

use nalgebra::DVector;

// TODO: Make some sort of macro for defining activation functions

pub struct Activation {
    pub apply: fn(DVector<Float>) -> DVector<Float>,
    pub derive: fn(DVector<Float>) -> DVector<Float>,
}

pub const SIGMOID: Activation = Activation {
    apply: |z| z.map(|x| sigmoid(x)),
    derive: |z| z.map(|x| d_sigmoid(x)),
};

pub const RELU: Activation = Activation {
    apply: |z| z.map(|x| relu(x)),
    derive: |z| z.map(|x| d_relu(x)),
};

fn sigmoid(x: Float) -> Float {
    use std::f32::consts::E;
    1.0 / (1.0 + E.powf(-x))
}

fn d_sigmoid(x: Float) -> Float {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn relu(x: Float) -> Float {
    0_f32.max(x)
}

fn d_relu(x: Float) -> Float {
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: Float = 0.000001;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < EPSILON);
        assert!((sigmoid(2.0) - 0.880797).abs() < EPSILON);
        assert!((sigmoid(-2.0) - 0.119202).abs() < EPSILON);

        assert!((d_sigmoid(0.0) - 0.25).abs() < EPSILON);
        assert!((d_sigmoid(2.0) - 0.104993).abs() < EPSILON);
        assert!((d_sigmoid(-2.0) - 0.104993).abs() < EPSILON);
    }

    #[test]
    fn test_relu() {
        assert!((relu(2.0) - 2.0).abs() < EPSILON);
        assert!((relu(0.0)).abs() < EPSILON);
        assert!((relu(-2.0)).abs() < EPSILON);

        assert!((d_relu(2.0) - 1.0).abs() < EPSILON);
        assert!((d_relu(0.0)).abs() < EPSILON);
        assert!((d_relu(-1.0)).abs() < EPSILON);
    }
}
