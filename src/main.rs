mod app;
mod data;
mod error;
mod model;
mod ui;

use app::app::App;
use error::VibeError;

fn main() -> Result<(), VibeError> {
    let mut app = App::new()?;
    app.run()?;

    Ok(())
}
