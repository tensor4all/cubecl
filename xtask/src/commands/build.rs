use tracel_xtask::prelude::*;

#[macros::extend_command_args(BuildCmdArgs, Target, None)]
pub struct CubeCLBuildCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: CubeCLBuildCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    if args.ci {
        args.exclude.extend(vec![
            "t4a-cubecl-cuda".to_string(),
            "t4a-cubecl-hip".to_string(),
            "t4a-cubecl-metal".to_string(),
        ]);
    }
    base_commands::build::handle_command(args.try_into().unwrap(), env, context)?;
    // Additional feature-specific builds
    build_helpers::custom_crates_build(
        vec!["t4a-cubecl-wgpu"],
        vec!["--features", "spirv"],
        None,
        None,
        "std with SPIR-V compiler",
    )?;
    build_helpers::custom_crates_build(
        vec!["t4a-cubecl-runtime", "t4a-cubecl-server"],
        vec!["--no-default-features"],
        None,
        None,
        "without default features",
    )?;

    Ok(())
}
