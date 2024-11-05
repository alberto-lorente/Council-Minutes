# Datapolitics Project

The aim of the project is to create a detector for "projects" implemented by local authorities based on the documents they produce.

We focus here on geothermal energy.

The material consists of a set (approximately 20,000) of PDF documents produced over the past 5 years by these local authorities and published on their websites. These are mostly minutes of municipal or regional council meetings, but the document type is not filtered a priori. The documents were extracted from the Datapolitics database based on the presence of the keyword "geothermal" (which is relatively unambiguous).

As a reference for currently implemented projects and to familiarize oneself with the field, one can consult the website [geothermies](https://www.geothermies.fr). The site particularly mentions the acronyms of projects or installations, which can be valuable in an exploration phase to trace them in the documents.

The objective is to provide (automatically), for each document, its status in relation to a geothermal project:

At a first level, a binary filter on the documents is requested:

- concerns a geothermal project
- unrelated to a geothermal project

At a second level, for those that pass the first filter, the following stages are distinguished:

- idea/wish,
- preliminary studies,
- budget voted for the definitive project,
- implementation in progress,
- installation completed

Final level of extraction: project data:

- initial budget
- final cost
- estimated duration
- actual duration

The deliverable will consist of a processing chain allowing for filtering and information extraction. The form is not constrained (scripts, Notebooks, etc.)
Care will be taken to ensure that the methodology implemented can be applied to all types of local projects (wind power, construction of health centers, renovation of public lighting, etc.)

# Dataset

The dataset is described by the `dataset.csv` file. The URLs link to the source version of the documents, but they are all available in cache. A version converted to plain text is also available.

The file schema is as follows:

| Field       | Content                                                     |
|-------------|-------------------------------------------------------------|
| doc_id      | Document identifier (internal to Datapolitics)              |
| url         | source url                                                  |
| cache       | link to the cached version of the PDF                       |
| fulltext    | link to the version converted to text                       |
| nature      | Document type (automatically calculated)                    |
| published   | Publication date (automatically calculated)                 |
| entity_name | Name of the local entity                                    |
| entity_type | type of entity (municipality, intercommunality, prefecture, etc.) |
| geo_path    | "path" in the administrative structure                      |

The `geo_path` field allows "ascending" the administrative levels. These levels are separated by "/". Typically, for a municipality, it will be: `name of the municipality/name of the intercommunality/name of the department/name of the region`. For higher levels, the path will be shorter.

The document types are produced by an automatic classifier. The nomenclature is as follows:

| Code               | Meaning                                |
|--------------------|----------------------------------------|
| acte.delib         | Deliberation                           |
| acte.arrete        | Decree                                 |
| acte.raa           | Collection of administrative acts      |
| pv.full            | Minutes                                |
| pv.cr              | Minutes                                |
| pv.odj             | Agenda                                 |
| pv.video           | Video                                  |
| projet             | Draft act                              |
| dlao.plu           | Local Urban Plan                       |
| dlao.pcaet         | Territorial Climate-Air-Energy Plan    |
| dlao.scot          | Territorial Coherence Scheme           |
| dlao.autres        | Other planning document                |
| mp.annonce         | Public procurement announcement        |
| mp.avenant         | Public procurement amendment           |
| mp.reglement       | Public procurement regulations         |
| comm               | Other communication                    |
| comm.org           | Organizational chart                   |
| rapport            | Report                                 |
| bdj                | Budget                                 |
| bdj.annexes        | Annexed budget document                |
| comm.agenda        | Agenda                                 |
| comm.contact       | Contact                                |
| comm.concert       | Consultation                           |
| comm.demarches     | Procedures                             |
| comm.emploi        | Employment                             |
| comm.present       | Presentation                           |
| comm.project       | Project                                |
| comm.magazine      | Periodical                             |
| comm.election      | Elections                              |
| comm.services      | Services                               |
| other              | Other                                  |
