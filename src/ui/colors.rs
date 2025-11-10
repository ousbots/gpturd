use ratatui::style::Color;

pub struct Palette {}

impl Palette {
    pub const FG_COLOR: Color = Color::Rgb(253, 255, 182); // #FDFFB6
    pub const BG_COLOR: Color = Color::Rgb(30, 30, 46); // #1E1E2E
    pub const BORDER_COLOR: Color = Color::Rgb(255, 214, 165); // #FFD6A5

    pub const TRAINING_LOSS_COLOR: Color = Color::Rgb(202, 255, 191); // #CAFFBF
    pub const VALIDATION_LOSS_COLOR: Color = Color::Rgb(189, 178, 255); // #BDB2FF
}

pub enum Rainbow {
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    Indigo,
    Violet,
}

impl Rainbow {
    const RED_GRADIENT: [u8; 6] = [41, 43, 50, 68, 104, 156];
    const GREEN_GRADIENT: [u8; 6] = [24, 30, 41, 65, 105, 168];
    const BLUE_GRADIENT: [u8; 6] = [55, 57, 62, 78, 113, 166];
    const AMBIENT_GRADIENT: [u8; 6] = [17, 18, 20, 25, 40, 60];

    pub const ROYGBIV: [Self; 7] = [
        Self::Red,
        Self::Orange,
        Self::Yellow,
        Self::Green,
        Self::Blue,
        Self::Indigo,
        Self::Violet,
    ];

    pub fn gradient_color(&self, row: usize) -> Color {
        let ambient = Self::AMBIENT_GRADIENT[row];
        let red = Self::RED_GRADIENT[row];
        let green = Self::GREEN_GRADIENT[row];
        let blue = Self::BLUE_GRADIENT[row];
        let blue_sat = Self::AMBIENT_GRADIENT[row].saturating_mul(6 - row as u8);
        let (r, g, b) = match self {
            Self::Red => (red, ambient, blue_sat),
            Self::Orange => (red, green / 2, blue_sat),
            Self::Yellow => (red, green, blue_sat),
            Self::Green => (ambient, green, blue_sat),
            Self::Blue => (ambient, ambient, blue.max(blue_sat)),
            Self::Indigo => (blue, ambient, blue.max(blue_sat)),
            Self::Violet => (red, ambient, blue.max(blue_sat)),
        };
        Color::Rgb(r, g, b)
    }
}
