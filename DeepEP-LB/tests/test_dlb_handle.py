"""Validate exact-size moved-only DLB dispatch handles."""

from dlb_test_extension import load_dlb_test_extension


def main() -> None:
    """Round-trip representative incoming and outgoing Rail mappings."""
    extension = load_dlb_test_extension()
    cases = (
        # Rail 0 receives 1500 moved tokens.
        (8000, 8000, 1500, 0, 2, 10, 49368),
        # Rail 3 sends 1500 moved tokens.
        (8000, 8000, 0, 1500, 2, 10, 40368),
        # A Rail can simultaneously send and receive moved tokens.
        (8000, 8000, 500, 500, 2, 10, 41368),
    )
    for case in cases:
        *arguments, expected_bytes = case
        handle, mismatches = extension.run_dlb_handle_roundtrip(
            *arguments
        )
        assert handle.numel() == expected_bytes
        assert mismatches.cpu().item() == 0
    print("compact moved-only handle round-trip passed")


if __name__ == "__main__":
    main()
