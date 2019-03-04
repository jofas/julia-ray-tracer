cargo web build --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/*.wasm www/
cp target/wasm32-unknown-unknown/release/*.js www/
