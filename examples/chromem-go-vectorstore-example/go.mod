module github.com/tmc/langchaingo/examples/chromem-go-vectorstore-example

go 1.22.0

require (
	// github.com/tmc/langchaingo v0.1.10
	github.com/tmc/langchaingo v0.0.0
)

require (
	github.com/philippgille/chromem-go v0.6.0 // indirect
	github.com/dlclark/regexp2 v1.10.0 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/pkoukk/tiktoken-go v0.1.6 // indirect
)

replace github.com/tmc/langchaingo => ../..

replace github.com/philippgille/chromem-go => /path/to/local/chromem-go
