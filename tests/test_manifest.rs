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
fn test_manifest_creation_v2() {
    let mut server = Server::new(); // Create a new mock server
    let claim_generator = "c2pa-ml/0.1.0".to_string();

    let file_path = "tests/test_data/model.bin";
    let file_content = fs::read(file_path).expect("Failed to read test file");

    // Mock the server to expect a GET request to /model.file
    let mock = server
        .mock("GET", "/model.file")
        .with_status(200)
        .with_header("content-type", "application/octet-stream")
        .with_body(file_content.clone())
        .create();

    let ingredient_data = IngredientData {
        url: server.url() + "/model.file",
        alg: "sha256".to_string(),
        hash: "".to_string(),
        data_types: vec![AssetType::ModelOpenVino],
        linked_ingredient_hash: None,
        linked_ingredient_url: None,
    };

    println!("URL being requested: {}", ingredient_data.url); // Print the actual URL
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(&ingredient_data.url)
        .send()
        .expect("Failed to make request");

    assert_eq!(response.status(), 200); // Ensure the status is OK
    let body = response.bytes().expect("Failed to read response body");

    let hash_value = sha256(&body);
    let hash_hex = hex::encode(hash_value);

    let expected_hash = "4e4fc7c4587ee5d5fa73f0b679e2d3549b5b0101fc556df783445a6db8b4161f";
    assert_eq!(hash_hex, expected_hash);

    mock.assert();

    let ingredient = Ingredient {
        title: "Ingredient 1".to_string(),
        format: "application/octet-stream".to_string(),
        relationship: "componentOf".to_string(),
        document_id: "uuid:87d51599-286e-43b2-9478-88c79f49c347".to_string(),
        instance_id: "uuid:7b57930e-2f23-47fc-affe-0400d70b738d".to_string(),
        data: ingredient_data,
        linked_ingredient: None,
        public_key: None,
    };

    let creative_work_assertion = Assertion::CreativeWork(CreativeWorkAssertion {
        context: "http://schema.org/".to_string(),
        creative_type: "CreativeWork".to_string(), // Correct field name
        author: vec![Author {
            author_type: "Person".to_string(),
            name: "John Doe".to_string(),
        }],
    });

    let action_assertion = Assertion::Action(ActionAssertion {
        actions: vec![Action {
            action: "c2pa.created".to_string(),
            software_agent: Some("c2pa-ml 0.1.0".to_string()),
            parameters: Some(serde_json::json!({
                "name": "Model 1",
                "version": "1.0",
            })),
            digital_source_type: Some(
                "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia".to_string(),
            ),
            instance_id: None,
        }],
    });

    let claim_v2 = ClaimV2 {
        instance_id: "xmp:iid:3d6ce559-af88-444c-808a-1e3ece74d175".to_string(),
        ingredients: vec![ingredient],
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
        ingredients: vec![],
        claim: claim_v2,
        cross_references: vec![],
        title: "Manifest Title".to_string(),
        is_active: false,
    };

    assert_eq!(manifest.claim_generator, "c2pa-ml/0.1.0");
    assert_eq!(
        manifest.claim_v2.as_ref().unwrap().instance_id,
        "xmp:iid:3d6ce559-af88-444c-808a-1e3ece74d175"
    );
    assert_eq!(manifest.claim_v2.as_ref().unwrap().ingredients.len(), 1);
    assert_eq!(
        manifest.claim_v2.as_ref().unwrap().created_assertions.len(),
        2
    );
    assert_eq!(
        manifest.claim_v2.as_ref().unwrap().signature,
        Some("Ps256".to_string())
    );
}
