use atlas_c2pa_lib::assertion::{
    Action, ActionAssertion, Assertion, Author, CreativeWorkAssertion,
};
use atlas_c2pa_lib::asset_type::AssetType;
use atlas_c2pa_lib::claim::ClaimV2;
use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
use atlas_c2pa_lib::ingredient::{Ingredient, IngredientData};
use atlas_c2pa_lib::manifest::Manifest;
use mockito::Server;
use openssl::sha::sha256;
use std::fs;
use time::OffsetDateTime;

#[test]
fn test_dataset_manifest_creation_v2() {
    let mut server = Server::new();
    let claim_generator = "c2pa-ml/0.1.0".to_string();

    let file_paths = [
        "tests/test_data/t10k-images-idx3-ubyte.gz",
        "tests/test_data/t10k-labels-idx1-ubyte.gz",
        "tests/test_data/train-labels-idx1-ubyte.gz",
    ];

    let mut ingredients = vec![];
    let mut mocks = vec![]; // Store mocks to keep them alive

    for (i, file_path) in file_paths.iter().enumerate() {
        let file_content = fs::read(file_path).expect("Failed to read test file");

        let mock = server
            .mock("GET", format!("/dataset_file_{i}").as_str())
            .with_status(200)
            .with_header("content-type", "application/octet-stream")
            .with_body(file_content.clone())
            .create();

        let ingredient_data = IngredientData {
            url: server.url() + &format!("/dataset_file_{i}"),
            alg: "sha256".to_string(),
            hash: "".to_string(),
            data_types: vec![AssetType::Dataset],
            linked_ingredient_hash: None,
            linked_ingredient_url: None,
        };

        println!("URL being requested: {}", ingredient_data.url); // Print URL
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(&ingredient_data.url)
            .send()
            .expect("Failed to make request");

        assert_eq!(response.status(), 200); // Ensure the status is OK
        let body = response.bytes().expect("Failed to read response body");

        let hash_value = sha256(&body);
        let hash_hex = hex::encode(hash_value);

        // Compare calculated hash with expected hash
        let expected_hash = match i {
            0 => "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
            1 => "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
            2 => "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
            _ => panic!("Unexpected file index"),
        };
        assert_eq!(hash_hex, expected_hash);

        mock.assert(); // Ensure the mock was hit

        let ingredient = Ingredient {
            title: format!("Ingredient {}", i + 1),
            format: "application/octet-stream".to_string(),
            relationship: "componentOf".to_string(),
            document_id: format!("uuid:dataset-file-{}", i + 1),
            instance_id: format!("uuid:dataset-instance-{}", i + 1),
            data: ingredient_data,
            linked_ingredient: None,
            public_key: None,
        };

        ingredients.push(ingredient);
        mocks.push(mock); // Keep mock alive
    }

    let creative_work_assertion = Assertion::CreativeWork(CreativeWorkAssertion {
        context: "http://schema.org/".to_string(),
        creative_type: "Dataset".to_string(),
        author: vec![Author {
            author_type: "Organization".to_string(),
            name: "MNIST".to_string(),
        }],
    });

    let action_assertion = Assertion::Action(ActionAssertion {
        actions: vec![Action {
            action: "c2pa.created".to_string(),
            software_agent: Some("c2pa-ml 0.1.0".to_string()),
            parameters: Some(serde_json::json!({
                "name": "MNIST Dataset",
                "version": "1.0",
            })),
            digital_source_type: Some(
                "http://cv.iptc.org/newscodes/digitalsourcetype/dataset".to_string(),
            ),
            instance_id: None,
        }],
    });

    let claim_v2 = ClaimV2 {
        instance_id: "xmp:iid:mnist-dataset-instance".to_string(),
        ingredients: ingredients.clone(),
        created_assertions: vec![creative_work_assertion, action_assertion],
        claim_generator_info: "Ps256".to_string(),
        signature: Some("Ps256".to_string()),
        created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
    };

    let manifest = Manifest {
        claim_generator: claim_generator.clone(),
        claim_v2: Some(claim_v2.clone()),
        created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
        instance_id: "xmp:iid:mnist-dataset-instance".to_string(),
        ingredients,
        title: "MNIST Dataset".to_string(),
        cross_references: vec![],
        claim: claim_v2,
        is_active: false,
    };

    // Assertions
    assert_eq!(manifest.claim_generator, "c2pa-ml/0.1.0");
    assert_eq!(manifest.instance_id, "xmp:iid:mnist-dataset-instance");
    assert_eq!(manifest.ingredients.len(), 3);
    assert_eq!(
        manifest.claim_v2.as_ref().unwrap().created_assertions.len(),
        2
    );
    assert_eq!(
        manifest.claim_v2.as_ref().unwrap().signature,
        Some("Ps256".to_string())
    );
}
