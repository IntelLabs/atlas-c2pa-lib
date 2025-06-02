use atlas_c2pa_lib::assertion::{
    Action, ActionAssertion, Assertion, Author, CreativeWorkAssertion,
};
use atlas_c2pa_lib::asset_type::AssetType;
use atlas_c2pa_lib::claim::ClaimV2;
use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
use atlas_c2pa_lib::ingredient::{Ingredient, IngredientData};
use atlas_c2pa_lib::manifest::{CrossReference, Manifest};
use mockito::mock;
use openssl::sha::sha256;
use time::OffsetDateTime;

#[test]
fn test_cross_manifest_linking() {
    let claim_generator = "c2pa-ml/0.1.0".to_string();

    let fixed_timestamp = OffsetDateTimeWrapper(
        OffsetDateTime::parse(
            "2024-09-23T12:34:56Z",
            &time::format_description::well_known::Rfc3339,
        )
        .unwrap(),
    );

    let linked_manifest = Manifest {
        claim_generator: claim_generator.clone(),
        title: "Linked Manifest".to_string(),
        instance_id: "xmp:iid:linked-manifest-12345".to_string(),
        ingredients: vec![],
        claim: ClaimV2 {
            instance_id: "xmp:iid:claim-v2-12345".to_string(),
            ingredients: vec![],
            created_assertions: vec![],
            claim_generator_info: "c2pa-ml".to_string(),
            signature: Some("dummy_signature".to_string()), // Fixed signature
            created_at: fixed_timestamp.clone(),
        },
        claim_v2: None,
        created_at: fixed_timestamp.clone(),
        cross_references: vec![],
        is_active: false,
    };

    let linked_manifest_json = serde_json::to_string(&linked_manifest).unwrap();

    let linked_manifest_hash = hex::encode(sha256(linked_manifest_json.as_bytes()));

    let linked_manifest_mock = mock("GET", "/linked_manifest.json")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(linked_manifest_json.clone()) // Fixed linked manifest
        .create();

    let ingredient_data = IngredientData {
        url: mockito::server_url() + "/model.file",
        alg: "sha256".to_string(),
        hash: "4e4fc7c4587ee5d5fa73f0b679e2d3549b5b0101fc556df783445a6db8b4161f".to_string(),
        data_types: vec![AssetType::ModelOpenVino],
        linked_ingredient_hash: None,
        linked_ingredient_url: None,
    };

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
        creative_type: "CreativeWork".to_string(),
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

    let manifest = Manifest {
        claim_generator: claim_generator.clone(),
        title: "Manifest Title".to_string(),
        instance_id: "xmp:iid:mnist-dataset-instance".to_string(),
        ingredients: vec![ingredient],
        claim: ClaimV2 {
            instance_id: "xmp:iid:claim-v2-12345".to_string(),
            ingredients: vec![],
            created_assertions: vec![creative_work_assertion.clone(), action_assertion.clone()],
            claim_generator_info: "c2pa-ml".to_string(),
            signature: Some("dummy_signature".to_string()),
            created_at: fixed_timestamp.clone(),
        },
        claim_v2: Some(ClaimV2 {
            instance_id: "xmp:iid:claim-v2-12345".to_string(),
            ingredients: vec![],
            created_assertions: vec![creative_work_assertion, action_assertion],
            claim_generator_info: "c2pa-ml".to_string(),
            signature: Some("dummy_signature".to_string()),
            created_at: fixed_timestamp.clone(),
        }),
        created_at: fixed_timestamp.clone(),
        cross_references: vec![CrossReference {
            manifest_url: mockito::server_url() + "/linked_manifest.json",
            manifest_hash: linked_manifest_hash.clone(),
            media_type: Some("application/json".to_string()),
        }],
        is_active: false,
    };

    let manifest_json = serde_json::to_string(&manifest).unwrap();
    let _manifest_hash = hex::encode(sha256(manifest_json.as_bytes()));

    println!("CrossReference Hash: {linked_manifest_hash}");

    let client = reqwest::blocking::Client::new();
    let response = client
        .get(&manifest.cross_references[0].manifest_url)
        .send()
        .expect("Failed to request linked manifest");

    assert_eq!(response.status(), 200);
    let body = response.text().expect("Failed to read response body");
    println!("Response Body: {body}");

    let known_manifest_hash = "8337441bdec617f12215056d9440f35fe3162fa569482f6454d4ccf5cc17d473";
    assert_eq!(
        linked_manifest_hash, known_manifest_hash,
        "Hash does not match the expected value"
    );

    linked_manifest_mock.assert();
}
