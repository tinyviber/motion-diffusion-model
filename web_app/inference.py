import json
import shutil
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

import torch

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from sample.generate import construct_template_variables
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.sampler_util import ClassifierFreeSampleModel
from visualize.motions2hik import motions2hik


def _default_args() -> Namespace:
    args = Namespace()
    args.fps = 20
    args.model_path = "./save/humanml_trans_enc_512/model000200000.pt"
    args.guidance_param = 2.5
    args.unconstrained = False
    args.dataset = "humanml"

    args.cond_mask_prob = 1
    args.emb_trans_dec = False
    args.latent_dim = 512
    args.layers = 8
    args.arch = "trans_enc"

    args.noise_schedule = "cosine"
    args.sigma_small = True
    args.lambda_vel = 0.0
    args.lambda_rcxyz = 0.0
    args.lambda_fc = 0.0
    return args


class MotionGenerator:
    """Utility class that mirrors the Cog predictor for interactive use."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        guidance_param: Optional[float] = None,
        output_dir: str = "web_app/outputs",
    ) -> None:
        args = _default_args()
        if model_path:
            args.model_path = model_path
        if guidance_param is not None:
            args.guidance_param = guidance_param

        self.args = args
        self.num_frames = self.args.fps * 6
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_model()

    def _prepare_clip_cache(self) -> None:
        clip_src = Path("ViT-B-32.pt")
        cache_dir = Path.home() / ".cache" / "clip"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / clip_src.name
        if clip_src.exists() and not cached.exists():
            shutil.copy(clip_src, cached)

    def _setup_model(self) -> None:
        self._prepare_clip_cache()
        print("Loading dataset...")
        data = get_dataset_loader(
            name=self.args.dataset,
            batch_size=1,
            num_frames=196,
            split="test",
            hml_mode="text_only",
        )
        data.fixed_length = float(self.num_frames)

        print("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(self.args, data)

        print("Loading checkpoints from...")
        state_dict = torch.load(self.args.model_path, map_location="cpu")
        load_model_wo_clip(model, state_dict)

        if self.args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)
        model.to(dist_util.dev())
        model.eval()

        self.model = model
        self.diffusion = diffusion
        self.data = data

    def _build_kwargs(self, prompt: str, num_repetitions: int) -> Dict:
        collate_args = [
            {
                "inp": torch.zeros(self.num_frames),
                "tokens": None,
                "lengths": self.num_frames,
                "text": str(prompt),
            }
        ]
        _, model_kwargs = collate(collate_args)
        if self.args.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(num_repetitions, device=dist_util.dev()) * self.args.guidance_param
            )
        return model_kwargs

    def _render_animations(
        self, prompt: str, all_motions, num_repetitions: int, out_root: Path
    ) -> List[Path]:
        caption = str(prompt)
        skeleton = paramUtil.t2m_kinematic_chain
        sample_print_template, row_print_template, all_print_template, sample_file_template, row_file_template, all_file_template = construct_template_variables(
            self.args.unconstrained
        )
        rep_files: List[Path] = []
        for rep_i in range(num_repetitions):
            motion = all_motions[rep_i].transpose(2, 0, 1)[: self.num_frames]
            save_file = out_root / sample_file_template.format(1, rep_i)
            print(sample_print_template.format(caption, 1, rep_i, save_file))
            plot_3d_motion(
                str(save_file),
                skeleton,
                motion,
                dataset=self.args.dataset,
                title=caption,
                fps=self.args.fps,
            )
            rep_files.append(save_file)
        return rep_files

    def _recover_motion(self, sample: torch.Tensor):
        if self.model.data_rep == "hml_vec":
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = self.data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = "xyz" if self.model.data_rep in ["xyz", "hml_vec"] else self.model.data_rep
        rot2xyz_mask = None
        if rot2xyz_pose_rep != "xyz":
            rot2xyz_mask = (
                self.model_kwargs["y"]["mask"].reshape(self.args.num_repetitions, self.num_frames).bool()
            )

        sample = self.model.rot2xyz(
            x=sample,
            mask=rot2xyz_mask,
            pose_rep=rot2xyz_pose_rep,
            glob=True,
            translation=True,
            jointstype="smpl",
            vertstrans=True,
            betas=None,
            beta=0,
            glob_rot=None,
            get_rotations_back=False,
        )
        return sample.cpu().numpy()

    def generate(
        self, prompt: str, num_repetitions: int = 3, output_format: str = "animation"
    ) -> Dict[str, object]:
        self.args.num_repetitions = int(num_repetitions)
        self.data = get_dataset_loader(
            name=self.args.dataset,
            batch_size=self.args.num_repetitions,
            num_frames=self.num_frames,
            split="test",
            hml_mode="text_only",
        )
        self.model_kwargs = self._build_kwargs(prompt, self.args.num_repetitions)

        sample_fn = self.diffusion.p_sample_loop
        sample = sample_fn(
            self.model,
            (self.args.num_repetitions, self.model.njoints, self.model.nfeats, self.num_frames),
            clip_denoised=False,
            model_kwargs=self.model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        all_motions = self._recover_motion(sample)
        run_dir = self.output_dir / f"run_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "json_file":
            data_dict = motions2hik(all_motions)
            json_path = run_dir / "motion.json"
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(data_dict, json_file)
            return {"message": "Generated JSON motion data", "animations": [], "json_file": json_path}

        rep_files = self._render_animations(prompt, all_motions, self.args.num_repetitions, run_dir)
        return {"message": "Generated animations", "animations": rep_files, "json_file": None}
