use atlas_c2pa_lib::asset_type::AssetType;
use atlas_c2pa_lib::claim::ClaimV2;
use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
use atlas_c2pa_lib::ingredient::{Ingredient, IngredientData, LinkedIngredient};
use atlas_c2pa_lib::manifest::Manifest;
use mockito::mock;
use openssl::sha::sha256;
use time::OffsetDateTime;

#[test]
fn test_ingredient_linking_with_verification() {
    let claim_generator = "c2pa-ml/0.1.0".to_string();

    // Fixed timestamp for consistent hashing
    let fixed_timestamp = OffsetDateTimeWrapper(
        OffsetDateTime::parse(
            "2024-09-23T12:34:56Z",
            &time::format_description::well_known::Rfc3339,
        )
        .unwrap(),
    );

    // Linked ingredient (that itself is a reference to another dataset or model)
    let linked_ingredient = LinkedIngredient {
        url: mockito::server_url() + "/linked_ingredient.json",
        hash: "ab82708c91050a674c1b12e2d48f4b2dced1dd25b1132d3f59460ec39ecce664".to_string(), // We'll verify this later
        media_type: "application/json".to_string(),
    };

    // Main ingredient with a link to another ingredient
    let ingredient_data = IngredientData {
        url: mockito::server_url() + "/ingredient.json",
        alg: "sha256".to_string(),
        hash: "ingredient_hash".to_string(),
        data_types: vec![AssetType::ModelOpenVino],
        linked_ingredient_url: Some(linked_ingredient.url.clone()), // Linking to another ingredient
        linked_ingredient_hash: Some(linked_ingredient.hash.clone()),
    };

    let ingredient = Ingredient {
        title: "Main Ingredient".to_string(),
        format: "application/json".to_string(),
        relationship: "componentOf".to_string(),
        document_id: "ingredient-doc-123".to_string(),
        instance_id: "ingredient-instance-123".to_string(),
        data: ingredient_data,
        linked_ingredient: Some(linked_ingredient.clone()), // Reference to the linked ingredient
        public_key: None,
    };

    // Claim with the ingredient
    let claim_v2 = ClaimV2 {
        instance_id: "xmp:iid:claim-v2-12345".to_string(),
        ingredients: vec![ingredient.clone()],
        created_assertions: vec![],
        claim_generator_info: "c2pa-ml".to_string(),
        signature: Some("dummy_signature".to_string()), // Dummy signature for testing
        created_at: fixed_timestamp.clone(),
    };

    // Manifest that includes the claim and the main ingredient
    let manifest = Manifest {
        claim_generator: claim_generator.clone(),
        title: "Main Manifest".to_string(),
        instance_id: "xmp:iid:manifest-12345".to_string(),
        ingredients: vec![ingredient.clone()],
        claim: claim_v2.clone(),
        created_at: fixed_timestamp.clone(),
        cross_references: vec![],
        claim_v2: Some(claim_v2.clone()),
        is_active: false,
    };

    // Serialize the manifest to JSON
    let manifest_json = serde_json::to_string(&manifest).unwrap();

    // Compute the hash of the manifest for linking purposes
    let _manifest_hash = hex::encode(sha256(manifest_json.as_bytes()));

    // Mock the HTTP server to return the linked ingredient
    let linked_ingredient_mock = mock("GET", "/linked_ingredient.json")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{ \"ingredient\": \"linked\" }") // Simulated linked ingredient
        .create();

    // Mock the HTTP server to return the ingredient
    let ingredient_mock = mock("GET", "/ingredient.json")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(manifest_json.clone()) // Return the serialized manifest
        .create();

    // Request the linked ingredient via HTTP mock
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(ingredient.data.linked_ingredient_url.unwrap())
        .send()
        .expect("Failed to request linked ingredient");

    // Assert the response is correct
    assert_eq!(response.status(), 200);
    let body = response.text().expect("Failed to read response body");

    // Verify the hash of the linked ingredient
    let computed_hash = hex::encode(sha256(body.as_bytes()));
    assert_eq!(
        computed_hash, linked_ingredient.hash,
        "Linked ingredient hash does not match!"
    );

    // Validate the HTTP mock was called
    linked_ingredient_mock.assert();

    // Request the main ingredient (with its linked ingredient)
    let main_ingredient_response = client
        .get(&ingredient.data.url)
        .send()
        .expect("Failed to request main ingredient");

    // Assert the response is correct for the main ingredient
    assert_eq!(main_ingredient_response.status(), 200);
    let main_body = main_ingredient_response
        .text()
        .expect("Failed to read main ingredient body");

    // Print the main body for debugging purposes
    println!("Main Ingredient Response Body: {main_body}");

    // Validate the main ingredient mock was called
    ingredient_mock.assert();
}
