const fs = require('fs');
const path = require('path');

const MODEL_JSON = 'C:\\Users\\Lenovo LOQ\\xPawSure-Capstone\\mobile\\assets\\model\\model.json';
const BACKUP_JSON = 'C:\\Users\\Lenovo LOQ\\xPawSure-Capstone\\mobile\\assets\\model\\model_before_flat.json';

// Keras 3 fields to strip from layer dicts
const KERAS3_STRIP_FIELDS = new Set([
  'module', 'registered_name', 'build_config', 'compile_config',
  'ragged', 'optional', 'synchronized', 'quantization_config'
]);
const REGULARIZER_FIELDS = new Set([
  'kernel_regularizer', 'bias_regularizer', 'activity_regularizer',
  'kernel_constraint', 'bias_constraint', 'beta_regularizer',
  'gamma_regularizer', 'beta_constraint', 'gamma_constraint'
]);
const BN_KERAS3_FIELDS = new Set(['renorm', 'renorm_clipping', 'renorm_momentum']);
const ALL_STRIP_FIELDS = new Set([...KERAS3_STRIP_FIELDS, ...REGULARIZER_FIELDS, ...BN_KERAS3_FIELDS]);

function convertInitializer(initCfg) {
  if (!initCfg || typeof initCfg !== 'object') return initCfg;
  const cn = initCfg.class_name || '';
  const mapping = {
    'GlorotUniform': 'glorot_uniform', 'GlorotNormal': 'glorot_normal',
    'HeUniform': 'he_uniform', 'HeNormal': 'he_normal',
    'LecunUniform': 'lecun_uniform', 'LecunNormal': 'lecun_normal',
    'Orthogonal': 'orthogonal', 'RandomNormal': 'RandomNormal',
    'RandomUniform': 'RandomUniform', 'Zeros': 'zeros', 'Ones': 'ones',
    'Identity': 'identity',
  };
  if (mapping[cn]) return mapping[cn];
  if (cn === 'VarianceScaling') {
    const cfg = initCfg.config || {};
    const scale = cfg.scale ?? 1.0;
    const mode = cfg.mode || 'fan_in';
    const dist = cfg.distribution || 'uniform';
    if (scale === 2.0 && mode === 'fan_out' && dist === 'truncated_normal') return 'he_normal';
    if (scale === 1.0 && mode === 'fan_in' && dist === 'uniform') return 'glorot_uniform';
    return 'glorot_uniform';
  }
  return 'glorot_uniform';
}

function cleanConfig(cfg) {
  if (!cfg || typeof cfg !== 'object') return cfg;
  const cleaned = {};
  for (const [k, v] of Object.entries(cfg)) {
    if (ALL_STRIP_FIELDS.has(k)) continue;
    if (Array.isArray(v)) {
      cleaned[k] = v;
    } else if (v && typeof v === 'object') {
      if (v.class_name && v.module) {
        cleaned[k] = convertInitializer(v);
      } else {
        const subCleaned = {};
        for (const [sk, sv] of Object.entries(v)) {
          if (!KERAS3_STRIP_FIELDS.has(sk)) subCleaned[sk] = sv;
        }
        cleaned[k] = subCleaned;
      }
    } else {
      cleaned[k] = v;
    }
  }
  return cleaned;
}

function stripInboundRefs4thElement(layersList) {
  for (const ld of layersList) {
    if (ld.inbound_nodes) {
      for (let i = 0; i < ld.inbound_nodes.length; i++) {
        const node = ld.inbound_nodes[i];
        ld.inbound_nodes[i] = node.map(ref => {
          if (Array.isArray(ref) && ref.length > 3) return ref.slice(0, 3);
          return ref;
        });
      }
    }
  }
}

// Read current model.json
console.log('Reading model.json...');
const modelJson = JSON.parse(fs.readFileSync(MODEL_JSON, 'utf8'));
const topology = modelJson.modelTopology;
const config = topology.config;

// Backup
fs.writeFileSync(BACKUP_JSON, JSON.stringify(modelJson, null, 0));
console.log('Backed up to model_before_flat.json');

const outerLayers = config.layers;
console.log(`Outer layers: ${outerLayers.length}`);
outerLayers.forEach(l => console.log(`  ${l.name} (${l.class_name})`));

// Find the Functional (efficientnetb0) layer
const effnetIdx = outerLayers.findIndex(l => l.class_name === 'Functional');
if (effnetIdx === -1) {
  console.error('No Functional layer found!');
  process.exit(1);
}
const effnet = outerLayers[effnetIdx];
const innerLayers = effnet.config.layers;
console.log(`\nInner EfficientNetB0 layers: ${innerLayers.length}`);

// Collect inner layer names for reference checking
const innerLayerNames = new Set(innerLayers.map(l => l.name));
console.log('Inner layer names (first 10):', [...innerLayerNames].slice(0, 10));

// Strip Keras 3 fields from inner layers
for (const ld of innerLayers) {
  for (const k of KERAS3_STRIP_FIELDS) delete ld[k];
  if (ld.config) ld.config = cleanConfig(ld.config);
}

// Strip trailing {} from inner inbound_nodes
stripInboundRefs4thElement(innerLayers);

// Build the flat layers list
// 1. Start with input_layer_1 (outer InputLayer)
// 2. Add all inner layers EXCEPT input_layer (inner InputLayer)
// 3. Add remaining outer layers (skipping the Functional wrapper)
const flatLayers = [];

// input_layer_1
flatLayers.push(outerLayers[0]); // input_layer_1

// Inner layers (excluding input_layer)
let skippedInner = 0;
for (const ld of innerLayers) {
  if (ld.class_name === 'InputLayer') {
    skippedInner++;
    continue; // Skip inner input_layer - redundant with input_layer_1
  }
  flatLayers.push(ld);
}
console.log(`\nSkipped ${skippedInner} inner InputLayer(s)`);

// Remaining outer layers (skip input_layer_1[0] and efficientnetb0[effnetIdx])
for (let i = 0; i < outerLayers.length; i++) {
  if (i === 0 || i === effnetIdx) continue;
  flatLayers.push(outerLayers[i]);
}

console.log(`\nFlat layers total: ${flatLayers.length}`);

// Fix inbound_nodes:
// 1. rescaling: ["input_layer",0,0] → ["input_layer_1",0,0]
// 2. global_average_pooling2d: ["efficientnetb0",0,0] → ["top_activation",0,0]
console.log('\nFixing inbound_nodes...');
let fixes = 0;
for (const ld of flatLayers) {
  if (!ld.inbound_nodes) continue;
  for (const node of ld.inbound_nodes) {
    for (const ref of node) {
      if (Array.isArray(ref)) {
        if (ref[0] === 'input_layer') {
          console.log(`  ${ld.name}: input_layer → input_layer_1`);
          ref[0] = 'input_layer_1';
          fixes++;
        } else if (ref[0] === 'efficientnetb0') {
          console.log(`  ${ld.name}: efficientnetb0 → top_activation`);
          ref[0] = 'top_activation';
          fixes++;
        }
      }
    }
  }
}
console.log(`Fixed ${fixes} inbound_nodes references`);

// Strip efficientnetb0/ prefix from weight names
console.log('\nStripping efficientnetb0/ prefix from weight names...');
const weights = modelJson.weightsManifest[0].weights;
let stripped = 0;
let noPrefix = 0;
for (const w of weights) {
  if (w.name.startsWith('efficientnetb0/')) {
    w.name = w.name.substring('efficientnetb0/'.length);
    stripped++;
  } else {
    noPrefix++;
  }
}
console.log(`Stripped prefix from ${stripped} weights, ${noPrefix} unchanged`);

// Verify weight names match layer names
console.log('\nVerifying weight names...');
const layerNames = new Set(flatLayers.map(l => l.name));
let mismatches = 0;
for (const w of weights) {
  const layerName = w.name.split('/')[0];
  if (!layerNames.has(layerName)) {
    console.log(`  MISMATCH: weight "${w.name}" has no matching layer "${layerName}"`);
    mismatches++;
  }
}
console.log(`Weight name mismatches: ${mismatches}`);

// Update the model topology
config.layers = flatLayers;

// Update input_layers and output_layers (ensure proper format)
config.input_layers = [['input_layer_1', 0, 0]];
config.output_layers = [['softmax', 0, 0]];
console.log(`\ninput_layers: ${JSON.stringify(config.input_layers)}`);
console.log(`output_layers: ${JSON.stringify(config.output_layers)}`);

// Remove the 'module' and 'registered_name' from topology if present
delete topology.module;
delete topology.registered_name;
delete topology.build_config;
delete topology.compile_config;

// Write the new model.json
console.log('\nWriting model.json...');
fs.writeFileSync(MODEL_JSON, JSON.stringify(modelJson));
console.log('Done!');

// Summary
console.log('\n=== SUMMARY ===');
console.log(`Flat layers: ${flatLayers.length}`);
console.log(`Nested Functional models: 0`);
console.log(`Weight specs: ${weights.length}`);
console.log(`Weight name mismatches: ${mismatches}`);
console.log(`Inbound_nodes fixes: ${fixes}`);
console.log(`Input layers: ${JSON.stringify(config.input_layers)}`);
console.log(`Output layers: ${JSON.stringify(config.output_layers)}`);
