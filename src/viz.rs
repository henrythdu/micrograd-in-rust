use plotly::common::{Marker, Mode, Title};
use plotly::layout::{Axis, AxisConstrain, Layout};
use plotly::{Plot, Scatter};

pub fn visualize_data(x_data: &Vec<Vec<f64>>, y_data: &Vec<f64>, title_str: &str) {
    let mut plot = Plot::new();

    let mut x_pos = Vec::new();
    let mut y_pos = Vec::new();
    let mut x_neg = Vec::new();
    let mut y_neg = Vec::new();

    for (coords, &label) in x_data.iter().zip(y_data.iter()) {
        if label > 0.0 {
            x_pos.push(coords[0]);
            y_pos.push(coords[1]);
        } else {
            x_neg.push(coords[0]);
            y_neg.push(coords[1]);
        }
    }

    let trace_pos = Scatter::new(x_pos, y_pos)
        .mode(Mode::Markers)
        .marker(Marker::new().color("maroon").size(10))
        .name("Class +1");

    let trace_neg = Scatter::new(x_neg, y_neg)
        .mode(Mode::Markers)
        .marker(Marker::new().color("darkslategrey").size(10))
        .name("Class -1");

    plot.add_trace(trace_pos);
    plot.add_trace(trace_neg);

    // Use the same range as visualize_boundary for consistency
    let (x_min, x_max) = (-1.5, 2.5);
    let (y_min, y_max) = (-1.2, 1.2);

    let layout = Layout::new()
        .title(Title::with_text(title_str))
        .width(800)
        .height(800)
        .x_axis(
            Axis::new()
                .range(vec![x_min, x_max])
                .scale_anchor("y")
                .constrain(AxisConstrain::Domain),
        )
        .y_axis(Axis::new().range(vec![y_min, y_max]));

    plot.set_layout(layout);
    plot.show();
}
