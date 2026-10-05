"""Build what the LaTeX template shows, with every string already escaped.

Two sources:
- a TailoredCV (the normal path): only selected entries and their reviewed bullets;
- the whole profile (`full_profile_cv`): every entry and bullet, for previewing the template.
Education and certifications always come from the profile; they are not tailored.
"""

from dataclasses import dataclass, field

from factfit.render.latex import format_range, tex_escape, tex_url
from factfit.schemas.profile import Experience, Profile, Project
from factfit.schemas.tailored import TailoredBullet, TailoredCV, TailoredSection, TailoredSummary


@dataclass
class EntryView:
    title: str  # role, or linked project name with its tech
    dates: str
    subtitle: str = ""  # organisation
    place: str = ""
    bullets: list[str] = field(default_factory=list)


@dataclass
class CVView:
    name: str
    headline: str
    contact: str  # "phone $|$ \href{mailto:...}{...} $|$ ..."
    summary: str
    skill_groups: list[tuple[str, str]]  # (heading, comma-separated skills)
    experiences: list[EntryView]
    projects: list[EntryView]
    education: list[EntryView]
    certifications: str


def full_profile_cv(profile: Profile) -> TailoredCV:
    """Every entry and bullet as written, for previewing the template."""
    sections = [
        TailoredSection(
            type=kind,
            ref=e.id,
            bullets=[
                TailoredBullet(text=b.text, source_bullet_ids=[b.id], grounding="pass")
                for b in e.bullets
            ],
        )
        for kind, entries in (("experience", profile.experiences), ("project", profile.projects))
        for e in entries
    ]
    summary = profile.summary_variants[0] if profile.summary_variants else None
    return TailoredCV(
        summary=TailoredSummary(text=summary.text, source_ids=[summary.id]) if summary else None,
        sections=sections,
        skills=[s.name for s in profile.skills],
    )


def build_view(profile: Profile, cv: TailoredCV) -> CVView:
    b = profile.basics
    contact = []
    if b.phone:
        contact.append(tex_escape(b.phone))
    contact.append(rf"\href{{mailto:{tex_url(b.email)}}}{{\underline{{{tex_escape(b.email)}}}}}")
    for value in (b.github, b.linkedin, b.website):
        if value:
            url = value if value.startswith("http") else f"https://{value}"
            shown = value.removeprefix("https://").removeprefix("http://")
            contact.append(rf"\href{{{tex_url(url)}}}{{\underline{{{tex_escape(shown)}}}}}")
    if b.location:
        contact.append(tex_escape(b.location))

    entries = {e.id: e for e in [*profile.experiences, *profile.projects]}
    experiences = [_experience(entries[s.ref], s) for s in cv.sections if s.type == "experience"]
    projects = [_project(entries[s.ref], s) for s in cv.sections if s.type == "project"]

    return CVView(
        name=tex_escape(b.name),
        headline=tex_escape(b.headline.upper()) if b.headline else "",
        contact=" $|$ ".join(contact),
        summary=tex_escape(cv.summary.text) if cv.summary else "",
        skill_groups=_skill_groups(profile, cv.skills),
        experiences=experiences,
        projects=projects,
        education=[
            EntryView(
                title=tex_escape(ed.school),
                place=tex_escape(ed.location or ""),
                dates=format_range(ed.start, ed.end),
                subtitle=tex_escape(ed.degree) + (f", GPA {tex_escape(ed.gpa)}" if ed.gpa else ""),
            )
            for ed in profile.education
        ],
        certifications=r" $\quad\vert\quad$ ".join(
            rf"\textbf{{{tex_escape(c.name)}}}" + (f" ({tex_escape(c.date)})" if c.date else "")
            for c in profile.certifications
        ),
    )


def _experience(entry: Experience, section: TailoredSection) -> EntryView:
    return EntryView(
        title=tex_escape(entry.role),
        dates=format_range(entry.start, entry.end),
        subtitle=tex_escape(entry.org),
        place=tex_escape(entry.location or ""),
        bullets=[tex_escape(b.text) for b in section.bullets],
    )


def _project(entry: Project, section: TailoredSection) -> EntryView:
    name = tex_escape(entry.name)
    if entry.url:
        url = entry.url if entry.url.startswith("http") else f"https://{entry.url}"
        name = rf"\href{{{tex_url(url)}}}{{{name}}}"
    tech = entry.tech or list(dict.fromkeys(s for b in entry.bullets for s in b.skills))
    title = rf"\textbf{{{name}}}"
    if tech:
        title += rf" $|$ \emph{{{tex_escape(', '.join(tech))}}}"
    return EntryView(
        title=title,
        dates=format_range(entry.start, entry.end),
        bullets=[tex_escape(b.text) for b in section.bullets],
    )


def _skill_groups(profile: Profile, names: list[str]) -> list[tuple[str, str]]:
    """Skills on this CV, grouped by category in the order categories first appear."""
    wanted = set(names)
    groups: dict[str, list[str]] = {}
    for s in profile.skills:
        if s.name in wanted:
            groups.setdefault(s.category or "Skills", []).append(s.name)
    return [(tex_escape(k), tex_escape(", ".join(v))) for k, v in groups.items()]
