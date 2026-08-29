// Package metrics parses the bounded Prometheus subset needed by benchmark evidence.
package metrics

import (
	"bufio"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// Series is one labeled Prometheus sample.
type Series struct {
	Name   string
	Labels map[string]string
	Value  float64
}

// Snapshot is indexed by a stable name-and-label key.
type Snapshot map[string]Series

// Parse accepts Prometheus text exposition samples and ignores comments.
func Parse(raw string) (Snapshot, error) {
	snapshot := make(Snapshot)
	scanner := bufio.NewScanner(strings.NewReader(raw))
	scanner.Buffer(make([]byte, 64*1024), 4<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		series, err := parseSample(line)
		if err != nil {
			return nil, err
		}
		snapshot[key(series.Name, series.Labels)] = series
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return snapshot, nil
}

func parseSample(line string) (Series, error) {
	space := strings.IndexAny(line, " \t")
	if space < 1 {
		return Series{}, fmt.Errorf("invalid Prometheus sample")
	}
	metricPart := line[:space]
	fields := strings.Fields(line[space:])
	if len(fields) == 0 {
		return Series{}, fmt.Errorf("missing Prometheus sample value")
	}
	value, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return Series{}, fmt.Errorf("invalid Prometheus value: %w", err)
	}
	series := Series{Labels: make(map[string]string), Value: value}
	brace := strings.IndexByte(metricPart, '{')
	if brace < 0 {
		series.Name = metricPart
		return series, nil
	}
	if !strings.HasSuffix(metricPart, "}") {
		return Series{}, fmt.Errorf("invalid Prometheus labels")
	}
	series.Name = metricPart[:brace]
	labels, err := parseLabels(metricPart[brace+1 : len(metricPart)-1])
	if err != nil {
		return Series{}, err
	}
	series.Labels = labels
	return series, nil
}

func parseLabels(raw string) (map[string]string, error) {
	labels := make(map[string]string)
	for index := 0; index < len(raw); {
		for index < len(raw) && (raw[index] == ' ' || raw[index] == ',') {
			index++
		}
		if index == len(raw) {
			break
		}
		start := index
		for index < len(raw) && raw[index] != '=' {
			index++
		}
		if index == len(raw) || index+1 >= len(raw) || raw[index+1] != '"' {
			return nil, fmt.Errorf("invalid Prometheus label")
		}
		name := raw[start:index]
		index += 2
		var value strings.Builder
		for index < len(raw) {
			if raw[index] == '"' {
				index++
				break
			}
			if raw[index] == '\\' {
				index++
				if index >= len(raw) {
					return nil, fmt.Errorf("invalid Prometheus label escape")
				}
				switch raw[index] {
				case 'n':
					value.WriteByte('\n')
				case '\\', '"':
					value.WriteByte(raw[index])
				default:
					return nil, fmt.Errorf("invalid Prometheus label escape")
				}
				index++
				continue
			}
			value.WriteByte(raw[index])
			index++
		}
		labels[name] = value.String()
	}
	return labels, nil
}

func key(name string, labels map[string]string) string {
	names := make([]string, 0, len(labels))
	for name := range labels {
		names = append(names, name)
	}
	sort.Strings(names)
	var builder strings.Builder
	builder.WriteString(name)
	for _, label := range names {
		builder.WriteByte('|')
		builder.WriteString(label)
		builder.WriteByte('=')
		builder.WriteString(labels[label])
	}
	return builder.String()
}

// Has reports whether any series exists for an exact metric name.
func (snapshot Snapshot) Has(name string) bool {
	for _, series := range snapshot {
		if series.Name == name {
			return true
		}
	}
	return false
}

// Sum returns all label-series values for a metric name.
func (snapshot Snapshot) Sum(name string) (float64, bool) {
	var sum float64
	found := false
	for _, series := range snapshot {
		if series.Name == name {
			sum += series.Value
			found = true
		}
	}
	return sum, found
}

// Delta is a before/after counter result. Reset makes Value invalid.
type Delta struct {
	Value   float64 `json:"value"`
	Found   bool    `json:"found"`
	Reset   bool    `json:"reset"`
	Missing bool    `json:"missing"`
}

// CounterDelta sums labeled counter deltas. A newly materialized after-series
// has an implicit zero before value; a disappeared after-series is missing.
func CounterDelta(before, after Snapshot, name string, labels map[string]string) Delta {
	beforeByLabels := matching(before, name, labels)
	afterByLabels := matching(after, name, labels)
	if len(afterByLabels) == 0 {
		return Delta{Missing: true}
	}
	result := Delta{Found: true}
	for seriesKey, afterValue := range afterByLabels {
		beforeValue := beforeByLabels[seriesKey]
		if afterValue < beforeValue {
			result.Reset = true
			continue
		}
		result.Value += afterValue - beforeValue
	}
	for seriesKey := range beforeByLabels {
		if _, exists := afterByLabels[seriesKey]; !exists {
			result.Missing = true
		}
	}
	return result
}

func matching(snapshot Snapshot, name string, labels map[string]string) map[string]float64 {
	result := make(map[string]float64)
	for _, series := range snapshot {
		if series.Name != name || !containsLabels(series.Labels, labels) {
			continue
		}
		identityLabels := make(map[string]string, len(series.Labels))
		for label, value := range series.Labels {
			identityLabels[label] = value
		}
		result[key("", identityLabels)] = series.Value
	}
	return result
}

func containsLabels(actual, wanted map[string]string) bool {
	for name, value := range wanted {
		if actual[name] != value {
			return false
		}
	}
	return true
}

// HistogramMean derives sum_delta/count_delta and detects either counter reset.
func HistogramMean(before, after Snapshot, base string) (*float64, bool) {
	sum := CounterDelta(before, after, base+"_sum", nil)
	count := CounterDelta(before, after, base+"_count", nil)
	if sum.Reset || count.Reset {
		return nil, true
	}
	if !sum.Found || !count.Found || count.Value <= 0 {
		return nil, false
	}
	value := sum.Value / count.Value
	return &value, false
}

// HistogramPercentile estimates a percentile at the first Prometheus bucket
// whose cumulative delta reaches nearest-rank ceil(p*count). It does not
// interpolate observations hidden inside buckets.
func HistogramPercentile(before, after Snapshot, base string, p float64) (*float64, bool) {
	type bucket struct {
		upper float64
		count float64
	}
	var buckets []bucket
	reset := false
	seen := make(map[float64]struct{})
	for _, series := range after {
		if series.Name != base+"_bucket" {
			continue
		}
		upper, err := strconv.ParseFloat(series.Labels["le"], 64)
		if err != nil {
			continue
		}
		if _, exists := seen[upper]; exists {
			continue
		}
		seen[upper] = struct{}{}
		delta := CounterDelta(before, after, base+"_bucket", map[string]string{"le": series.Labels["le"]})
		if delta.Reset {
			reset = true
		}
		if delta.Found {
			buckets = append(buckets, bucket{upper: upper, count: delta.Value})
		}
	}
	if reset || len(buckets) == 0 {
		return nil, reset
	}
	sort.Slice(buckets, func(i, j int) bool { return buckets[i].upper < buckets[j].upper })
	total := buckets[len(buckets)-1].count
	if total <= 0 {
		return nil, false
	}
	rank := math.Ceil(p * total)
	lastFinite := 0.0
	for _, item := range buckets {
		if !math.IsInf(item.upper, 1) {
			lastFinite = item.upper
		}
		if item.count >= rank {
			if math.IsInf(item.upper, 1) {
				return &lastFinite, false
			}
			value := item.upper
			return &value, false
		}
	}
	return nil, false
}
