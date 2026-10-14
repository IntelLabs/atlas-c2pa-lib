//! GPU vs CPU hashing benchmarks using Criterion

use atlas_c2pa_lib::gpu::{GpuHashAlgorithm, GpuHasher};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ring::digest::{Context, SHA256, SHA384, SHA512};

fn cpu_hash_sha256(data: &[u8]) -> Vec<u8> {
    let mut ctx = Context::new(&SHA256);
    ctx.update(data);
    ctx.finish().as_ref().to_vec()
}

fn cpu_hash_batch_sha256(messages: &[Vec<u8>]) -> Vec<Vec<u8>> {
    messages.iter().map(|m| cpu_hash_sha256(m)).collect()
}

fn bench_single_cpu_vs_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_cpu_vs_gpu");

    let sizes = [64, 1024, 4096, 65536, 1024 * 1024];

    for size in sizes {
        let data = vec![0xABu8; size];
        let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256).unwrap();

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("cpu_ring", size), &data, |b, data| {
            b.iter(|| cpu_hash_sha256(data))
        });

        group.bench_with_input(BenchmarkId::new("gpu", size), &data, |b, data| {
            b.iter(|| hasher.hash(data).unwrap())
        });
    }

    group.finish();
}

fn bench_batch_cpu_vs_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cpu_vs_gpu");

    let message_size = 4096;
    let batch_sizes = [10, 100, 1000, 5000];

    for batch_size in batch_sizes {
        let messages: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| {
                let mut msg = vec![0u8; message_size];
                for (j, byte) in msg.iter_mut().enumerate() {
                    *byte = ((i + j) % 256) as u8;
                }
                msg
            })
            .collect();

        let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256).unwrap();
        let total_bytes = message_size * batch_size;
        let label = format!("{}x{}B", batch_size, message_size);

        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("cpu_ring", &label),
            &messages,
            |b, messages| b.iter(|| cpu_hash_batch_sha256(messages)),
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_batch", &label),
            &messages,
            |b, messages| b.iter(|| hasher.hash_batch(messages).unwrap()),
        );
    }

    group.finish();
}

fn bench_large_file_cpu_vs_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_file_cpu_vs_gpu");
    group.sample_size(10);

    let chunk_size = 4096;
    let num_chunks = 25600; // ~100MB

    let chunks: Vec<Vec<u8>> = (0..num_chunks)
        .map(|i| {
            let mut chunk = vec![0u8; chunk_size];
            for (j, byte) in chunk.iter_mut().enumerate() {
                *byte = ((i + j) % 256) as u8;
            }
            chunk
        })
        .collect();

    let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256).unwrap();
    let total_bytes = chunk_size * num_chunks;

    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("cpu_ring_100MB", |b| {
        b.iter(|| cpu_hash_batch_sha256(&chunks))
    });

    group.bench_function("gpu_batch_100MB", |b| {
        b.iter(|| hasher.hash_batch(&chunks).unwrap())
    });

    group.finish();
}

fn bench_algorithms_cpu_vs_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms_cpu_vs_gpu");

    let batch_size = 1000;
    let message_size = 4096;

    let messages: Vec<Vec<u8>> = (0..batch_size)
        .map(|_| vec![0xABu8; message_size])
        .collect();

    let total_bytes = message_size * batch_size;
    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("cpu_sha256", |b| {
        b.iter(|| cpu_hash_batch_sha256(&messages))
    });

    let hasher256 = GpuHasher::new(GpuHashAlgorithm::Sha256).unwrap();
    group.bench_function("gpu_sha256", |b| {
        b.iter(|| hasher256.hash_batch(&messages).unwrap())
    });

    group.bench_function("cpu_sha384", |b| {
        b.iter(|| {
            messages
                .iter()
                .map(|m| {
                    let mut ctx = Context::new(&SHA384);
                    ctx.update(m);
                    ctx.finish().as_ref().to_vec()
                })
                .collect::<Vec<_>>()
        })
    });

    let hasher384 = GpuHasher::new(GpuHashAlgorithm::Sha384).unwrap();
    group.bench_function("gpu_sha384", |b| {
        b.iter(|| hasher384.hash_batch(&messages).unwrap())
    });

    group.bench_function("cpu_sha512", |b| {
        b.iter(|| {
            messages
                .iter()
                .map(|m| {
                    let mut ctx = Context::new(&SHA512);
                    ctx.update(m);
                    ctx.finish().as_ref().to_vec()
                })
                .collect::<Vec<_>>()
        })
    });

    let hasher512 = GpuHasher::new(GpuHashAlgorithm::Sha512).unwrap();
    group.bench_function("gpu_sha512", |b| {
        b.iter(|| hasher512.hash_batch(&messages).unwrap())
    });

    group.finish();
}

fn bench_crossover_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover_point");

    let message_size = 4096;
    let batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    for batch_size in batch_sizes {
        let messages: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| {
                let mut msg = vec![0u8; message_size];
                for (j, byte) in msg.iter_mut().enumerate() {
                    *byte = ((i + j) % 256) as u8;
                }
                msg
            })
            .collect();

        let hasher = GpuHasher::new(GpuHashAlgorithm::Sha256).unwrap();
        let total_bytes = message_size * batch_size;

        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("cpu", batch_size),
            &messages,
            |b, messages| b.iter(|| cpu_hash_batch_sha256(messages)),
        );

        group.bench_with_input(
            BenchmarkId::new("gpu", batch_size),
            &messages,
            |b, messages| b.iter(|| hasher.hash_batch(messages).unwrap()),
        );
    }

    group.finish();
}

fn bench_4gb_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("4gb_file");
    group.sample_size(10);

    // 4GB as 1MB chunks (4096 chunks)
    let chunk_size = 1024 * 1024; // 1MB
    let num_chunks = 4096; // 4GB total

    eprintln!("Allocating 4GB test data ({} x 1MB chunks)...", num_chunks);

    let chunks: Vec<Vec<u8>> = (0..num_chunks)
        .map(|i| {
            let mut chunk = vec![0u8; chunk_size];
            for (j, byte) in chunk.iter_mut().enumerate() {
                *byte = ((i + j) % 256) as u8;
            }
            chunk
        })
        .collect();

    let total_bytes = (chunk_size * num_chunks) as u64; // 4GB
    group.throughput(Throughput::Bytes(total_bytes));

    eprintln!("Starting 4GB benchmark...");

    // SHA-256
    group.bench_function("cpu_sha256_4GB", |b| {
        b.iter(|| cpu_hash_batch_sha256(&chunks))
    });

    let hasher256 = GpuHasher::new(GpuHashAlgorithm::Sha256).unwrap();
    group.bench_function("gpu_sha256_4GB", |b| {
        b.iter(|| hasher256.hash_batch(&chunks).unwrap())
    });

    // SHA-384
    group.bench_function("cpu_sha384_4GB", |b| {
        b.iter(|| {
            chunks
                .iter()
                .map(|m| {
                    let mut ctx = Context::new(&SHA384);
                    ctx.update(m);
                    ctx.finish().as_ref().to_vec()
                })
                .collect::<Vec<_>>()
        })
    });

    let hasher384 = GpuHasher::new(GpuHashAlgorithm::Sha384).unwrap();
    group.bench_function("gpu_sha384_4GB", |b| {
        b.iter(|| hasher384.hash_batch(&chunks).unwrap())
    });

    // SHA-512
    group.bench_function("cpu_sha512_4GB", |b| {
        b.iter(|| {
            chunks
                .iter()
                .map(|m| {
                    let mut ctx = Context::new(&SHA512);
                    ctx.update(m);
                    ctx.finish().as_ref().to_vec()
                })
                .collect::<Vec<_>>()
        })
    });

    let hasher512 = GpuHasher::new(GpuHashAlgorithm::Sha512).unwrap();
    group.bench_function("gpu_sha512_4GB", |b| {
        b.iter(|| hasher512.hash_batch(&chunks).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_cpu_vs_gpu,
    bench_batch_cpu_vs_gpu,
    bench_large_file_cpu_vs_gpu,
    bench_algorithms_cpu_vs_gpu,
    bench_crossover_point,
);

criterion_group!(
    name = large_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_4gb_file
);

criterion_main!(benches, large_benches);
