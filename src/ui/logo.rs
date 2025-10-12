use crate::ui::colors::{Palette, Rainbow};

use ratatui::{
    layout::{Constraint, Flex, Layout, Margin, Rect},
    widgets::Block,
    Frame,
};
use std::iter::zip;

const LOGO_TOP: &str = "▄▀▀  █▀▀▄ ▀█▀ █  █ █▀▀▄ █▀▀▄";
const LOGO_BOT: &str = "▀▄▄▀ █▀▀   █  ▀▄▄▀ █▀▀▄ █▄▄▀";

pub fn render(frame: &mut Frame, area: Rect) {
    let area = area.inner(Margin::new(1, 0));
    let layout = Layout::vertical(Constraint::from_lengths([6, 1, 1])).flex(Flex::End);
    let [shadow_area, logo_top_area, logo_bottom_area] = area.layout(&layout);

    // Divide the logo into letter sections, rendering a block for each letter with a color based on row index.
    let letter_layout = Layout::horizontal(Constraint::from_lengths([5, 5, 4, 5, 5, 4]));
    for (row_index, row) in shadow_area.rows().enumerate() {
        for (rainbow, letter_area) in zip(Rainbow::ROYGBIV, row.layout_vec(&letter_layout)) {
            let color = rainbow.gradient_color(row_index);
            frame.render_widget(Block::new().style(color), letter_area);
        }
        // Render the logo truncated.
        frame.render_widget(LOGO_TOP, row);
    }

    frame.render_widget(Block::new().style(Palette::FG_COLOR), logo_top_area);
    frame.render_widget(Block::new().style(Palette::FG_COLOR), logo_bottom_area);
    frame.render_widget(LOGO_TOP, logo_top_area);
    frame.render_widget(LOGO_BOT, logo_bottom_area);
}
