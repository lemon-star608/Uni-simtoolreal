# Compatibility patches

The pristine Git object and SHA256 remain the provenance anchor for each entry. The patched SHA256 is the only accepted runtime byte sequence.

## `rl_games/algos_torch/players.py`

- Pristine Git blob: `e511462f11acdba00ac7d1af3b8730194b349761`
- Pristine SHA256: `10e71a5bf243460b5192ecc6b24e1c37d94a8886ef4977967b1b63b8702f8438`
- Patched SHA256: `bfbeda07fe71f179788f2984f65583a68e80066a6abb1372fb11ccb4ce85173c`
- Reason: Replace the eager legacy Gym import with Gymnasium for the native continuous player import closure.
- Covering test: `tests/algos/rlgames_sapg/test_import.py::test_native_runner_agent_central_value_builder_and_player_are_vendored`

## `rl_games/common/a2c_common.py`

- Pristine Git blob: `d3e778231faf5eac8ec5768c631b2ffd36fabaab`
- Pristine SHA256: `a94ac0bf09c6c2253784a0dc2b6508c4cf0d3ab04d86d71e3483250f082aaf54`
- Patched SHA256: `07b58e70bdb3634be6450e3f518e96d79ce0f5097f1363369f732383f2b27a6d`
- Reason: Replace the eager legacy Gym import with Gymnasium and terminate the previously unterminated Source file with a newline; tensor formulas and control flow are unchanged.
- Covering test: `tests/algos/rlgames_sapg/test_import.py::test_native_runner_agent_central_value_builder_and_player_are_vendored`

## `rl_games/common/env_configurations.py`

- Pristine Git blob: `01102be2ba85a219113ca43dac4aa88027a335c8`
- Pristine SHA256: `47fd976b412382ff5ca0f01488fbcccb2d9e610c3e8a5d4f13ca2ee9fe2bcdc7`
- Patched SHA256: `3587096d857addf865698e0de5c4ab960aeeaff14bf4fe25b859533ba4e5c879`
- Reason: Replace the eager legacy Gym and wrapper imports with Gymnasium and terminate the previously unterminated Source file with a newline; optional legacy environment factories retain their local Gym imports and registration logic is unchanged.
- Covering test: `tests/vendor/test_simtoolreal_rl_games_vendor.py::test_editable_install_does_not_pollute_vendor_inventory`

## `rl_games/common/experience.py`

- Pristine Git blob: `e447af4fd49447fe1084e00cde37ce1d1c1d026e`
- Pristine SHA256: `943afd12f34729972d3a35e12b7673f1a5bc96faf57b5065fefe56343a57b871`
- Patched SHA256: `4f120a84884884c85a600881513a4bc6d8e8ca93ff0b005ee4da3e0d32bdf6ee`
- Reason: Use Gymnasium spaces and NumPy's supported np.bool_ scalar alias in replay and action-mask buffers; shapes and dtypes are unchanged.
- Covering test: `tests/algos/rlgames_sapg/test_import.py::test_removed_numpy_bool_aliases_keep_replay_and_action_mask_dtypes`

## `rl_games/common/player.py`

- Pristine Git blob: `83162989a8217dbfb4b053e67fb7c07edb79a08b`
- Pristine SHA256: `0118466077c35e8f2e2209ff9083620b0b18a76d6e1ef677d115b3d28e059296`
- Patched SHA256: `2bfef880376ae518b301508ad6591cf22808b86819d69306aa3e6a91a0ef700a`
- Reason: Replace the eager legacy Gym import with Gymnasium for the native base-player import closure.
- Covering test: `tests/algos/rlgames_sapg/test_import.py::test_native_runner_agent_central_value_builder_and_player_are_vendored`

## `rl_games/common/vecenv.py`

- Pristine Git blob: `646da55527ec9344da2809b26fd5fee7c68ee74c`
- Pristine SHA256: `48d63fce4c7300e2dd4f4f3962236899c25e0f409c9470bc60a095e3d86444a3`
- Patched SHA256: `e8d61e25e13b661d3543d14eef8e57056dd915ca62f6fbd0c880c922d6e539c4`
- Reason: Replace the eager legacy Gym import with Gymnasium and terminate the previously unterminated Source file with a newline; vector-environment routing is unchanged.
- Covering test: `tests/vendor/test_simtoolreal_rl_games_vendor.py::test_editable_install_does_not_pollute_vendor_inventory`

## `rl_games/common/wrappers.py`

- Pristine Git blob: `a62e0855d6cf8f965f9d2f26ae8a1771bd7754de`
- Pristine SHA256: `b8091f49ac9233aabcfa18bb0f706ff3a66798c975205ea0afe1648083eb2569`
- Patched SHA256: `fd9424ef1b3b564978f58a6a607b5ed5cf29bd5def9ec82a4b39601c34a08a04`
- Reason: Replace legacy Gym and spaces imports with Gymnasium and remove one redundant trailing blank line; wrapper control flow is unchanged.
- Covering test: `tests/vendor/test_simtoolreal_rl_games_vendor.py::test_editable_install_does_not_pollute_vendor_inventory`
