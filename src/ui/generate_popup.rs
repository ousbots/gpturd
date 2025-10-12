use crate::ui::colors::Palette;

use ratatui::{
    layout::{Constraint, Flex, Layout},
    text::{Line, Span},
    widgets::{Block, BorderType, Padding, Paragraph},
    Frame,
};

pub fn draw(frame: &mut Frame, generated: &Vec<String>) {
    let area = frame.area();
    let vertical = Layout::vertical([Constraint::Percentage(60)]).flex(Flex::Center);
    let horizontal = Layout::horizontal([Constraint::Percentage(20)]).flex(Flex::Center);
    let [area] = area.layout(&vertical);
    let [area] = area.layout(&horizontal);

    let generated_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Palette::BORDER_COLOR)
        .padding(Padding::horizontal(1))
        .title("Generated Text");

    let lines: Vec<Line> = generated
        .iter()
        .rev()
        .map(|text| Line::from(vec![Span::raw(text)]))
        .collect();

    frame.render_widget(Paragraph::new(lines).block(generated_block), area);
}
