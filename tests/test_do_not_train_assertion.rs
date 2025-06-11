use atlas_c2pa_lib::assertion::Assertion;
use atlas_c2pa_lib::assertion::DoNotTrainAssertion;
use atlas_c2pa_lib::claim::ClaimV2;
use atlas_c2pa_lib::datetime_wrapper::OffsetDateTimeWrapper;
use atlas_c2pa_lib::manifest::Manifest;
use atlas_c2pa_lib::manifest::validate_manifest;
use time::OffsetDateTime;

#[cfg(test)]
mod tests {
    use super::*; // Import all the necessary modules, structs, and traits from the current file

    #[test]
    fn test_do_not_train_assertion_valid() {
        // Create a valid DoNotTrainAssertion
        let do_not_train = DoNotTrainAssertion {
            reason: "Sensitive data, not suitable for training.".to_string(),
            enforced: true,
        };

        // Assert that verification passes for a valid assertion
        let result = do_not_train.verify();
        assert!(
            result.is_ok(),
            "Expected valid DoNotTrainAssertion to pass verification"
        );
    }

    #[test]
    fn test_do_not_train_assertion_invalid_empty_reason() {
        // Create an invalid DoNotTrainAssertion with an empty reason
        let do_not_train = DoNotTrainAssertion {
            reason: "".to_string(),
            enforced: true,
        };

        // Assert that verification fails due to empty reason
        let result = do_not_train.verify();
        assert!(
            result.is_err(),
            "Expected invalid DoNotTrainAssertion to fail due to empty reason"
        );
        assert_eq!(
            result.unwrap_err(),
            "[DoNotTrainAssertion] Missing required reason field".to_string()
        );
    }

    #[test]
    fn test_do_not_train_assertion_not_enforced() {
        // Create an invalid DoNotTrainAssertion with enforced set to false
        let do_not_train = DoNotTrainAssertion {
            reason: "Confidential data.".to_string(),
            enforced: false,
        };

        // Assert that verification fails due to enforced being false
        let result = do_not_train.verify();
        assert!(
            result.is_err(),
            "Expected invalid DoNotTrainAssertion to fail due to not being enforced"
        );
        assert_eq!(
            result.unwrap_err(),
            "[DoNotTrainAssertion] Assertion must have enforced=true to be valid".to_string()
        );
    }

    #[test]
    fn test_manifest_with_do_not_train_assertion() {
        // Create a valid DoNotTrainAssertion
        let do_not_train = DoNotTrainAssertion {
            reason: "This data is confidential.".to_string(),
            enforced: true,
        };

        // Create a sample claim with the DoNotTrainAssertion
        let assertion = Assertion::DoNotTrain(do_not_train);
        let claim_v2 = ClaimV2 {
            instance_id: "xmp:iid:claim-v2-12345".to_string(),
            created_assertions: vec![assertion.clone()], // Add the assertion to the claim
            ingredients: vec![],
            signature: None,
            claim_generator_info: "c2pa-ml".to_string(),
            created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
        };

        // Create a sample manifest
        let manifest = Manifest {
            claim_generator: "c2pa-ml".to_string(),
            title: "Test Manifest".to_string(),
            instance_id: "xmp:iid:manifest-12345".to_string(),
            ingredients: vec![],
            claim: claim_v2.clone(),
            created_at: OffsetDateTimeWrapper(OffsetDateTime::now_utc()),
            cross_references: vec![],
            claim_v2: Some(claim_v2),
            is_active: true,
        };

        // Verify the manifest and ensure DoNotTrainAssertion is valid
        let result = validate_manifest(&manifest);
        assert!(
            result.is_ok(),
            "Expected manifest with valid DoNotTrainAssertion to pass verification"
        );
    }
}
