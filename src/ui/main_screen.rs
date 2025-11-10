use crate::app::{device, options::Options};
use crate::ui::{colors::Palette, generate_popup, logo};

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect, Spacing},
    style::{Color, Style, Stylize},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, BorderType, Chart, Dataset, GraphType, Padding, Paragraph},
};

// Draw the main screen showing the options and model training statistics with dynamic loss data.
pub fn draw(
    frame: &mut Frame,
    options: &Options,
    loss_data: &Vec<(f64, f64)>,
    validation_loss_data: &Vec<(f64, f64)>,
    generated: &Vec<String>,
    show_generate: bool,
) {
    let area = frame.area();

    frame.buffer_mut().set_style(area, (Palette::FG_COLOR, Palette::BG_COLOR));

    let main_layout = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]);
    let content_layout = Layout::horizontal([Constraint::Length(30), Constraint::Fill(1)]);

    let [title_area, main_area] = main_layout.areas(area);
    let [config_area, model_area] = content_layout.areas(main_area);

    let title = Block::new()
        .title_alignment(Alignment::Center)
        .title("Generative Pretrained Turd".bold());
    frame.render_widget(title, title_area);

    let options_lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("data=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.data.clone()),
        ]),
        Line::from(vec![
            Span::styled("device=", Style::default().fg(Color::Blue).bold()),
            if options.device == device::DEVICE_NAME_CPU {
                Span::styled(options.device.clone(), Style::default().bg(Color::Red).fg(Color::Yellow).bold())
            } else {
                Span::raw(options.device.clone())
            },
        ]),
        Line::from(vec![
            Span::styled("iterations=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.iterations.to_string()),
        ]),
        Line::from(vec![
            Span::styled("generate=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.generate.to_string()),
        ]),
    ];

    let parameters_lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("batch_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.batch_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("block_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.block_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("embedding_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.embedding_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("hidden_size=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.hidden_size.to_string()),
        ]),
        Line::from(vec![
            Span::styled("learn_rate=", Style::default().fg(Color::Blue).bold()),
            Span::raw(options.learn_rate.to_string()),
        ]),
    ];

    let config_layout = Layout::vertical([Constraint::Length(8), Constraint::Fill(1)]).spacing(Spacing::Space(1));
    let [logo_area, lower_config_area] = config_area.layout(&config_layout);

    let options_layout = Layout::vertical(Constraint::from_lengths([
        options_lines.len() as u16 + 2,
        parameters_lines.len() as u16 + 2,
    ]));
    let [options_area, parameters_area] = lower_config_area.layout(&options_layout);

    logo::render(frame, logo_area);

    let options_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Palette::BORDER_COLOR)
        .padding(Padding::horizontal(1))
        .title("Training Options");

    frame.render_widget(Paragraph::new(options_lines).block(options_block), options_area);

    let parameters_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Palette::BORDER_COLOR)
        .padding(Padding::horizontal(1))
        .title("Model Hyperparameters");

    frame.render_widget(Paragraph::new(parameters_lines).block(parameters_block), parameters_area);

    render_loss(frame, model_area, options, loss_data, validation_loss_data);

    if show_generate {
        generate_popup::draw(frame, generated);
    }
}

// Render the loss chart with dynamic data.
fn render_loss(frame: &mut Frame, area: Rect, options: &Options, loss_data: &[(f64, f64)], validation_loss_data: &[(f64, f64)]) {
    // Use either dynamic data or default data
    let training_data = loss_data.to_vec();

    let validation_data = validation_loss_data.to_vec();

    let max_x = if let Some(elem) = loss_data.last() {
        if elem.0 > options.iterations as f64 {
            (elem.0 / 100.).ceil() * 100.
        } else {
            options.iterations as f64
        }
    } else {
        options.iterations as f64
    };

    let max_y = 4.;
    let min_y = 2.;

    let datasets = vec![
        Dataset::default()
            .name("Training Loss")
            .marker(Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(Palette::TRAINING_LOSS_COLOR)
            .data(&training_data),
        Dataset::default()
            .name("Validation Loss")
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Palette::VALIDATION_LOSS_COLOR)
            .data(&validation_data),
    ];

    let x_labels = vec!["0".to_string(), max_x.to_string()];
    let y_labels = vec![min_y.to_string(), max_y.to_string()];

    let chart = Chart::new(datasets)
        .style(Style::default().fg(Palette::FG_COLOR).bg(Palette::BG_COLOR))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Palette::BORDER_COLOR)
                .style(Style::default().fg(Palette::FG_COLOR))
                .title(Line::from("Loss").cyan().bold().centered()),
        )
        .x_axis(
            Axis::default()
                .bounds([0., max_x])
                .style(Style::default().fg(Palette::FG_COLOR))
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .bounds([0., max_y])
                .style(Style::default().fg(Palette::FG_COLOR))
                .labels(y_labels),
        );

    frame.render_widget(chart, area);
}
