use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use anyhow::{Result, bail};
use byteorder::{BigEndian, ReadBytesExt};

use crate::types::DataVector;

pub struct MnistData {
    training_labels: Vec<u8>,
}

// First 2 bytes of magics are always 0, byte 3 is the data type
// This data set uses ubyte so it should be 0x08
// Byte 4 is the number of dimensions, for images it's 3 (images, rows, cols)
const IMG_DATASET_MAGIC: u32 = 0x0000_0803;

// For the labels, the above is the same, except it's one dimensional
const LABEL_DATASET_MAGIC: u32 = 0x0000_0801;

const TRAINING_DATA_LENGTH: usize = 60_000;
const TESTING_DATA_LENGTH: usize = 10_000;

impl MnistData {
    fn labels(label_file_path: PathBuf, expected_length: usize) -> Result<Vec<u8>> {
        // Open file
        let mut file = File::open(label_file_path)?;
        let magic = file.read_u32::<BigEndian>()?;
        if magic != LABEL_DATASET_MAGIC {
            bail!("Expected {IMG_DATASET_MAGIC}, got {magic}")
        }

        let data_length = file.read_u32::<BigEndian>()?;
        assert_eq!(data_length, expected_length as u32);
        Ok(file.bytes().map(|b| b.unwrap()).collect())
    }

    fn images(image_file_path: PathBuf, expected_length: usize) -> Result<Vec<u8>> {
        bail!("Not implemented")
    }

    pub fn from_files(label_file_path: PathBuf) -> Result<MnistData> {
        let labels = MnistData::labels(label_file_path, TRAINING_DATA_LENGTH)?;
        let _images = MnistData::images("".into(), TRAINING_DATA_LENGTH)?;
        Ok(MnistData {
            training_labels: labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};

    use crate::mnist_loader::MnistData;

    #[test]
    fn read_data() {
        let data_path: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "data",
            "train-labels-idx1-ubyte",
        ]
        .iter()
        .collect();

        println!("{}", data_path.to_string_lossy().to_string());

        {
            let file = File::open(&data_path);
            assert!(file.is_ok());
        }

        let data = MnistData::from_files(data_path);
        assert!(data.is_ok());
    }
}
