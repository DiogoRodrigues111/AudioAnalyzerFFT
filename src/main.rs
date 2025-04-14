use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{num_complex::Complex32, num_traits::ToPrimitive, FftPlanner};
use eframe::{self, egui, App};
use egui_plot::{self, Plot, Line};

const FFT_SIZE: usize = 1024;

fn main() -> eframe::Result<()> {
    let shared_data = Arc::new(Mutex::new(AudioData::new()));
    start_audio_stream(shared_data.clone());

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Rust Audio Analyzer",
        options,
        Box::new(|_cc| Ok(Box::new(AudioApp { shared_data }))),
    )
}

struct AudioApp {
    shared_data: Arc<Mutex<AudioData>>,
}

impl App for AudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let data = self.shared_data.lock().unwrap();

            ui.heading("🎧 Waveform");
            Plot::new("waveform")
                .height(150.0)
                .show(ui, |plot_ui| {
                    let points: Vec<_> = data.waveform
                        .iter()
                        .enumerate()
                        .map(|(i, &y)| [i as f64, y as f64])
                        .collect();
                    plot_ui.line(Line::new("points", points));
                });

            ui.separator();
            ui.heading("📊 FFT Spectrum");
            Plot::new("fft")
                .height(150.0)
                .show(ui, |plot_ui| {
                    let points: Vec<_> = data.spectrum
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| [i as f64, v as f64])
                        .collect();
                    plot_ui.line(Line::new("points", points));
                });
        });

        ctx.request_repaint(); // atualização contínua
    }
}

struct AudioData {
    waveform: Vec<f32>,
    spectrum: Vec<f32>,
}

impl AudioData {
    fn new() -> Self {
        Self {
            waveform: vec![0.0; FFT_SIZE],
            spectrum: vec![0.0; FFT_SIZE / 2],
        }
    }
}

fn start_audio_stream(shared_data: Arc<Mutex<AudioData>>) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device");
    let config = device.default_input_config().unwrap();

    let sample_format = config.sample_format();
    let config = config.into();

    std::thread::spawn(move || {
        match sample_format {
            cpal::SampleFormat::F32 => run_stream::<f32>(&device, &config, shared_data),
            cpal::SampleFormat::I16 => run_stream::<i16>(&device, &config, shared_data),
            cpal::SampleFormat::U16 => run_stream::<u16>(&device, &config, shared_data),
            _ => panic!("Unsupported sample format"),
        }
    });
}

fn run_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared_data: Arc<Mutex<AudioData>>,
) where
    T: cpal::Sample + cpal::SizedSample
{
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut index = 0;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _| {
            for &sample in data {
                buffer[index] = sample.to_f32().unwrap();
                index += 1;

                if index >= FFT_SIZE {
                    let mut input: Vec<Complex32> =
                        buffer.iter().map(|&x| Complex32::new(x, 0.0)).collect();
                    fft.process(&mut input);

                    let mut spectrum = vec![0.0f32; FFT_SIZE / 2];
                    for i in 0..FFT_SIZE / 2 {
                        spectrum[i] = (input[i].norm().log10() + 1.0) * 10.0;
                    }

                    let mut data = shared_data.lock().unwrap();
                    data.waveform.copy_from_slice(&buffer);
                    data.spectrum.copy_from_slice(&spectrum);
                    index = 0;
                }
            }
        },
        move |err| eprintln!("Stream error: {:?}", err),
        Some(std::time::Duration::from_secs(4)),
    ).unwrap();

    stream.play().unwrap();
    std::thread::park(); // mantém thread ativa
}
