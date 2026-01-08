mod engine;
use engine::Area;

fn main() {
    let mut area = Area::new();
    let a = area.scalar(2.0);
    let b = area.scalar(3.0);
    let mul = area.mul(a, b); // 6.0
    let res = area.pow(mul, 2.0); // 36.0

    area.backward(res);

    println!("Value: {}", area.get_value(res).data); // 36
    println!("Grad of a: {}", area.get_value(a).grad); // 36 2ab**2
    println!("Grad of b: {}", area.get_value(b).grad); // 24 2ba**2
}
