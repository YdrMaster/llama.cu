macro_rules! destruct {
        ([$( $name:ident ),+] = $iter:expr) => {
            let mut iter = $iter.into_iter();
            $( let $name = iter.next().unwrap(); )+
            assert!(iter.next().is_none());
        };
    }

macro_rules! dims {
    ($pat:pat = $tensor:expr) => {
        let &$pat = &*$tensor.shape() else {
            panic!("Ndim mismatch ( = {})", $tensor.shape().len())
        };
    };
}

macro_rules! strides {
    ($pat:pat = $tensor:expr) => {
        let &$pat = &*$tensor.strides() else {
            panic!("Ndim mismatch ( = {})", $tensor.strides().len())
        };
    };
}

macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}

macro_rules! meta {
    ($gguf:expr => $key:ident) => {
        $gguf.$key().unwrap()
    };
    ($gguf:expr => $key:ident; $default:expr) => {
        match $gguf.$key() {
            Ok(val) => val,
            Err(ggus::GGufMetaError::NotExist) => $default,
            Err(e) => panic!("failed to read meta: {e:?}"),
        }
    };

    ($gguf:expr => (usize) $key:expr) => {
        $gguf.get_usize($key).unwrap()
    };
    ($gguf:expr => (usize) $key:expr; $default:expr) => {
        match $gguf.get_usize($key) {
            Ok(val) => val,
            Err(ggus::GGufMetaError::NotExist) => $default,
            Err(e) => panic!("failed to read meta: {e:?}"),
        }
    };
}

pub(crate) use {destruct, dims, meta, print_now, strides};
